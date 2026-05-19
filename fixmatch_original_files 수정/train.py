import argparse
import csv
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def get_class_groups(args):
    if hasattr(args, 'labeled_class_counts'):
        counts = list(args.labeled_class_counts)
        sorted_cls = sorted(range(args.num_classes),
                            key=lambda i: counts[i], reverse=True)
    else:
        sorted_cls = list(range(args.num_classes))
    group = max(1, args.num_classes // 3)
    head = sorted_cls[:group]
    mid = sorted_cls[group:2 * group]
    tail = sorted_cls[2 * group:]
    return head, mid, tail


def confusion_from_targets(targets, preds, num_classes):
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    targets = targets.detach().cpu().view(-1).long()
    preds = preds.detach().cpu().view(-1).long()
    for target, pred in zip(targets, preds):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            confusion[target, pred] += 1
    return confusion


def metrics_from_confusion(confusion, args):
    confusion = confusion.to(torch.float)
    true_hist = confusion.sum(dim=1)
    pred_hist = confusion.sum(dim=0)
    correct = torch.diag(confusion)
    total = true_hist.sum().clamp_min(1.0)

    per_class_acc = correct / true_hist.clamp_min(1.0) * 100.0
    precision = correct / pred_hist.clamp_min(1.0)
    recall = correct / true_hist.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    head, mid, tail = get_class_groups(args)

    return {
        'accuracy': (correct.sum() / total * 100.0).item(),
        'macro_f1': (f1.mean() * 100.0).item(),
        'head_accuracy': per_class_acc[head].mean().item(),
        'mid_accuracy': per_class_acc[mid].mean().item() if mid else 0.0,
        'tail_accuracy': per_class_acc[tail].mean().item() if tail else 0.0,
        'per_class_accuracy': per_class_acc.tolist(),
        'per_class_precision': (precision * 100.0).tolist(),
        'per_class_recall': (recall * 100.0).tolist(),
        'per_class_f1': (f1 * 100.0).tolist(),
        'prediction_histogram': pred_hist.long().tolist(),
        'true_histogram': true_hist.long().tolist(),
        'confusion_matrix': confusion.long().tolist(),
        'head_classes': head,
        'mid_classes': mid,
        'tail_classes': tail,
    }


def round_list(values, ndigits=2):
    return [round(float(v), ndigits) for v in values]


def ensure_metrics_dir(args):
    metrics_dir = os.path.join(args.out, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir


def save_confusion_csv(path, confusion):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred'] + list(range(len(confusion))))
        for class_idx, row in enumerate(confusion):
            writer.writerow([class_idx] + row)


def save_vector_csv(path, header, values):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class'] + header)
        for class_idx, row in enumerate(zip(*values)):
            writer.writerow([class_idx] + list(row))


def append_summary_csv(path, row):
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_metrics(prefix, metrics):
    logger.info("%s accuracy: %.2f", prefix, metrics['accuracy'])
    logger.info("%s macro_f1: %.2f", prefix, metrics['macro_f1'])
    logger.info("%s head/mid/tail acc: %.2f / %.2f / %.2f",
                prefix, metrics['head_accuracy'], metrics['mid_accuracy'],
                metrics['tail_accuracy'])
    logger.info("%s per-class acc: %s", prefix,
                round_list(metrics['per_class_accuracy']))
    logger.info("%s per-class f1: %s", prefix,
                round_list(metrics['per_class_f1']))
    logger.info("%s prediction histogram: %s", prefix,
                metrics['prediction_histogram'])
    logger.info("%s confusion matrix: %s", prefix,
                metrics['confusion_matrix'])


def save_eval_metrics(args, epoch, metrics):
    metrics_dir = ensure_metrics_dir(args)
    save_confusion_csv(
        os.path.join(metrics_dir, 'test_confusion_latest.csv'),
        metrics['confusion_matrix'])
    save_confusion_csv(
        os.path.join(metrics_dir, 'test_confusion_epoch_{:04d}.csv'.format(epoch + 1)),
        metrics['confusion_matrix'])
    save_vector_csv(
        os.path.join(metrics_dir, 'test_per_class_latest.csv'),
        ['accuracy', 'precision', 'recall', 'f1', 'true_count', 'pred_count'],
        [round_list(metrics['per_class_accuracy'], 6),
         round_list(metrics['per_class_precision'], 6),
         round_list(metrics['per_class_recall'], 6),
         round_list(metrics['per_class_f1'], 6),
         metrics['true_histogram'],
         metrics['prediction_histogram']])
    append_summary_csv(os.path.join(metrics_dir, 'test_summary.csv'), {
        'epoch': epoch + 1,
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'head_accuracy': metrics['head_accuracy'],
        'mid_accuracy': metrics['mid_accuracy'],
        'tail_accuracy': metrics['tail_accuracy'],
        'prediction_histogram': metrics['prediction_histogram'],
        'tail_classes': metrics['tail_classes'],
    })


def save_pseudo_metrics(args, epoch, all_metrics, selected_metrics,
                        selected_ratio):
    metrics_dir = ensure_metrics_dir(args)
    save_confusion_csv(
        os.path.join(metrics_dir, 'pseudo_all_confusion_latest.csv'),
        all_metrics['confusion_matrix'])
    save_confusion_csv(
        os.path.join(metrics_dir, 'pseudo_selected_confusion_latest.csv'),
        selected_metrics['confusion_matrix'])
    append_summary_csv(os.path.join(metrics_dir, 'pseudo_summary.csv'), {
        'epoch': epoch + 1,
        'all_accuracy': all_metrics['accuracy'],
        'all_macro_f1': all_metrics['macro_f1'],
        'all_tail_accuracy': all_metrics['tail_accuracy'],
        'selected_accuracy': selected_metrics['accuracy'],
        'selected_macro_f1': selected_metrics['macro_f1'],
        'selected_tail_accuracy': selected_metrics['tail_accuracy'],
        'selected_ratio': selected_ratio,
        'all_prediction_histogram': all_metrics['prediction_histogram'],
        'selected_prediction_histogram': selected_metrics['prediction_histogram'],
        'tail_classes': all_metrics['tail_classes'],
    })


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        pseudo_confusion_all = torch.zeros(args.num_classes, args.num_classes,
                                           dtype=torch.long)
        pseudo_confusion_selected = torch.zeros(args.num_classes, args.num_classes,
                                                dtype=torch.long)
        pseudo_selected_count = 0
        pseudo_total_count = 0
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), targets_u_true = next(unlabeled_iter)
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_u_true = next(unlabeled_iter)
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            targets_u_true = targets_u_true.to(args.device)
            with torch.no_grad():
                pseudo_confusion_all += confusion_from_targets(
                    targets_u_true, targets_u, args.num_classes)
                selected = mask.bool()
                if selected.any():
                    pseudo_confusion_selected += confusion_from_targets(
                        targets_u_true[selected], targets_u[selected],
                        args.num_classes)
                pseudo_selected_count += int(selected.sum().item())
                pseudo_total_count += int(targets_u_true.numel())

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_metrics = test(
                args, test_loader, test_model, epoch)
            pseudo_all_metrics = metrics_from_confusion(
                pseudo_confusion_all, args)
            pseudo_selected_metrics = metrics_from_confusion(
                pseudo_confusion_selected, args)
            pseudo_selected_ratio = pseudo_selected_count / max(
                1, pseudo_total_count)
            log_metrics('pseudo-all', pseudo_all_metrics)
            log_metrics('pseudo-selected', pseudo_selected_metrics)
            logger.info('pseudo selected ratio: %.4f', pseudo_selected_ratio)
            save_pseudo_metrics(args, epoch, pseudo_all_metrics,
                                pseudo_selected_metrics,
                                pseudo_selected_ratio)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.macro_f1',
                                   test_metrics['macro_f1'], epoch)
            args.writer.add_scalar('test/4.tail_acc',
                                   test_metrics['tail_accuracy'], epoch)
            args.writer.add_scalar('pseudo/1.all_acc',
                                   pseudo_all_metrics['accuracy'], epoch)
            args.writer.add_scalar('pseudo/2.all_macro_f1',
                                   pseudo_all_metrics['macro_f1'], epoch)
            args.writer.add_scalar('pseudo/3.all_tail_acc',
                                   pseudo_all_metrics['tail_accuracy'], epoch)
            args.writer.add_scalar('pseudo/4.selected_acc',
                                   pseudo_selected_metrics['accuracy'], epoch)
            args.writer.add_scalar('pseudo/5.selected_macro_f1',
                                   pseudo_selected_metrics['macro_f1'], epoch)
            args.writer.add_scalar('pseudo/6.selected_tail_acc',
                                   pseudo_selected_metrics['tail_accuracy'], epoch)
            args.writer.add_scalar('pseudo/7.selected_ratio',
                                   pseudo_selected_ratio, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'macro_f1': test_metrics['macro_f1'],
                'tail_acc': test_metrics['tail_accuracy'],
                'per_class_acc': test_metrics['per_class_accuracy'],
                'prediction_histogram': test_metrics['prediction_histogram'],
                'confusion_matrix': test_metrics['confusion_matrix'],
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    confusion = torch.zeros(args.num_classes, args.num_classes,
                            dtype=torch.long)
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            confusion += confusion_from_targets(targets, preds,
                                                args.num_classes)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    metrics = metrics_from_confusion(confusion, args)
    log_metrics('test', metrics)
    save_eval_metrics(args, epoch, metrics)
    return losses.avg, top1.avg, metrics


if __name__ == '__main__':
    main()
