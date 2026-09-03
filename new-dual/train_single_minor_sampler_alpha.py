import argparse
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar_val_split import DATASET_GETTERS
from train import (AverageMeter, PseudoLabelStats, append_csv_row, build_model,
                   checked_class_counts, classwise_topk_mask, count_parameters,
                   create_optimizer, get_cosine_schedule_with_warmup,
                   get_peak_memory, interleave, de_interleave, linear_rampup,
                   log_pseudo_group_scalars, minority_biased_probs,
                   ramped_class_weights, reconcile_labeled_class_counts,
                   reset_peak_memory, save_checkpoint, set_seed, test,
                   write_pseudo_stats, write_train_metrics)

logger = logging.getLogger(__name__)
best_acc = 0



def compute_per_class_metrics(confusion):
    confusion = confusion.float()
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    support = confusion.sum(dim=1)
    precision = tp / (tp + fp).clamp_min(1.0) * 100.0
    recall = tp / (tp + fn).clamp_min(1.0) * 100.0
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return precision, recall, f1, support


def write_eval_stats_with_precision(args, epoch, model_name, class_correct,
                                    class_total, confusion):
    per_class_acc = class_correct / class_total.clamp_min(1.0) * 100.0
    per_class_precision, per_class_recall, per_class_f1, support = \
        compute_per_class_metrics(confusion)
    macro_f1 = per_class_f1.mean().item()
    fields = ['epoch', 'model', 'class', 'accuracy', 'precision',
              'recall', 'f1', 'support']
    path = os.path.join(args.out, 'per_class_eval.csv')
    for class_idx in range(args.num_classes):
        append_csv_row(path, fields, {
            'epoch': epoch + 1,
            'model': model_name,
            'class': class_idx,
            'accuracy': per_class_acc[class_idx].item(),
            'precision': per_class_precision[class_idx].item(),
            'recall': per_class_recall[class_idx].item(),
            'f1': per_class_f1[class_idx].item(),
            'support': support[class_idx].item(),
        })
    return per_class_acc, macro_f1


def test(args, test_loader, model, epoch, model_name='minor'):
    losses = AverageMeter()
    top1 = AverageMeter()
    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
    confusion = torch.zeros(args.num_classes, args.num_classes)
    model.eval()
    loader = tqdm(test_loader, disable=args.no_progress)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            for class_idx in range(args.num_classes):
                class_mask = targets.eq(class_idx)
                class_total[class_idx] += class_mask.sum().cpu()
                class_correct[class_idx] += preds[class_mask].eq(
                    targets[class_mask]).sum().cpu()
            for true_class, pred_class in zip(targets.cpu(), preds.cpu()):
                confusion[true_class.long(), pred_class.long()] += 1
            acc = preds.eq(targets).float().mean().item() * 100.0
            losses.update(loss.item(), inputs.shape[0])
            top1.update(acc, inputs.shape[0])
    per_class, macro_f1 = write_eval_stats_with_precision(
        args, epoch, model_name, class_correct, class_total, confusion)
    logger.info('top-1 acc: %.2f', top1.avg)
    logger.info('per-class acc: %s', [round(v, 2) for v in per_class.tolist()])
    logger.info('macro F1: %.2f', macro_f1)
    if not hasattr(args, 'eval_macro_f1'):
        args.eval_macro_f1 = {}
    args.eval_macro_f1[model_name] = macro_f1
    return losses.avg, top1.avg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single minor-only FixMatch with balanced sampler and weighted CE')
    parser.add_argument('--gpu-id', default='0', type=int)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--num-labeled', type=int, default=4000)
    parser.add_argument('--expand-labels', action='store_true')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'])
    parser.add_argument('--total-steps', default=2**20, type=int)
    parser.add_argument('--eval-step', default=1024, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float)
    parser.add_argument('--warmup', default=0, type=float)
    parser.add_argument('--wdecay', default=5e-4, type=float)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--mu', default=7, type=int)
    parser.add_argument('--lambda-u', default=1, type=float)
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--imb-ratio', default=1.0, type=float)
    parser.add_argument('--val-samples-per-class', default=500, type=int)
    parser.add_argument('--imb-type', default='exp', choices=['exp', 'step'])
    parser.add_argument('--major-top-ratio', default=0.2, type=float)
    parser.add_argument('--classwise-quota', default='global',
                        choices=['global', 'batch'])
    parser.add_argument('--minority-threshold-gamma', default=1.0, type=float)
    parser.add_argument('--min-threshold', default=0.5, type=float)
    parser.add_argument('--minority-bias-strength', default=1.0, type=float)
    parser.add_argument('--minority-supervised-gamma', default=1.0, type=float)
    parser.add_argument('--minor-balanced-sampler-alpha', default=1.2, type=float,
                        help='Exponent for minor-only sampler weights: weight = 1 / class_count ** alpha. alpha > 1 biases more toward tail classes.')
    parser.add_argument('--max-minority-weight', default=10.0, type=float)
    parser.add_argument('--minority-bias-warmup', default=2048, type=int)
    parser.add_argument('--unsup-warmup', default=1024, type=int)
    parser.add_argument('--pseudo-warmup', default=512, type=int)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--init-checkpoint', default='', type=str)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--no-save-checkpoint', action='store_true')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    if args.no_ema:
        args.use_ema = False
    args.dual_train_mode = 'single_minor'
    args.use_minor_balanced_sampler = True
    args.use_minor_weighted_ce = True
    args.ensemble_minor_weight = 1.0
    return args


def setup_dataset_and_model(args):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.model_depth = 28
        args.model_width = 2
        args.model_cardinality = 4
    else:
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
            args.model_cardinality = 8
        else:
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64


def train_single_minor(args, labeled_loader, unlabeled_loader, test_loader,
                       model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    class_counts = checked_class_counts(
        torch.tensor(args.labeled_class_counts, dtype=torch.float), args)
    global_step = args.start_epoch * args.eval_step

    for epoch in range(args.start_epoch, args.epochs):
        reset_peak_memory(args)
        epoch_start = time.time()
        pseudo_stats = PseudoLabelStats(args.num_classes)
        model.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        p_bar = tqdm(range(args.eval_step), disable=args.no_progress)
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), targets_u_true = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s), targets_u_true = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                                2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_u_true = targets_u_true.to(args.device)

            logits = de_interleave(model(inputs), 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            bias_ramp = linear_rampup(global_step, args.minority_bias_warmup)
            sup_weight = ramped_class_weights(
                class_counts, args, bias_ramp).to(args.device)
            Lx = F.cross_entropy(logits_x, targets_x, weight=sup_weight,
                                 reduction='mean')

            probs = minority_biased_probs(logits_u_w, class_counts, args,
                                          bias_ramp)
            max_probs, targets_u = torch.max(probs, dim=-1)
            threshold_ramp = linear_rampup(global_step - args.pseudo_warmup,
                                           args.minority_bias_warmup)
            confidence_floor = args.threshold - \
                (args.threshold - args.min_threshold) * threshold_ramp
            mask = classwise_topk_mask(max_probs, targets_u, class_counts,
                                       args, confidence_floor)
            if global_step < args.pseudo_warmup:
                mask = torch.zeros_like(mask)

            pseudo_stats.update(targets_u, targets_u_true, mask)
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
            ramp = linear_rampup(global_step - args.pseudo_warmup,
                                 args.unsup_warmup)
            loss = Lx + ramp * args.lambda_u * Lu
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            global_step += 1

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    'Train Epoch: {}/{}. Iter: {}/{}. LR: {:.4f}. Loss: {:.4f}. '
                    'Loss_x: {:.4f}. Loss_u: {:.4f}. Mask: {:.2f}.'.format(
                        epoch + 1, args.epochs, batch_idx + 1, args.eval_step,
                        scheduler.get_last_lr()[0], losses.avg, losses_x.avg,
                        losses_u.avg, mask_probs.avg))
                p_bar.update()
        if not args.no_progress:
            p_bar.close()

        test_model = ema_model.ema if args.use_ema else model
        test_loss, test_acc = test(args, test_loader, test_model, epoch, 'minor')
        eval_f1 = getattr(args, 'eval_macro_f1', {})
        eval_f1['final'] = eval_f1.get('minor', '')
        peak_allocated, peak_reserved = get_peak_memory(args)
        write_pseudo_stats(args, epoch, 'minor', pseudo_stats)
        log_pseudo_group_scalars(args, 'minor', pseudo_stats, epoch)
        write_train_metrics(args, epoch, {
            'mode': 'single_minor',
            'epoch_time_sec': time.time() - epoch_start,
            'peak_allocated_mb': peak_allocated,
            'peak_reserved_mb': peak_reserved,
            'param_count_m': count_parameters(model),
            'param_count_major_m': 0,
            'param_count_minor_m': count_parameters(model),
            'train_loss': losses.avg,
            'train_loss_x': losses_x.avg,
            'train_loss_u': losses_u.avg,
            'mask': mask_probs.avg,
            'mask_minor': mask_probs.avg,
            'test_acc': test_acc,
            'test_acc_minor': test_acc,
            'test_acc_final': test_acc,
            'macro_f1': eval_f1.get('minor', ''),
            'macro_f1_minor': eval_f1.get('minor', ''),
            'macro_f1_final': eval_f1.get('minor', ''),
        })

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if not args.no_save_checkpoint:
            model_to_save = model.module if hasattr(model, 'module') else model
            ema_to_save = ema_model.ema.module if args.use_ema and hasattr(
                ema_model.ema, 'module') else (ema_model.ema if args.use_ema else None)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
        test_accs.append(test_acc)
        logger.info('Single minor top-1 acc: %.2f', test_acc)
        logger.info('Best single minor top-1 acc: %.2f', best_acc)
        logger.info('Mean single minor top-1 acc: %.2f\n',
                    np.mean(test_accs[-20:]))
    args.writer.close()


def main():
    args = parse_args()
    if args.local_rank == -1:
        args.device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() \
            else torch.device('cpu')
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning('Process rank: %s, device: %s, n_gpu: %s, distributed training: %s',
                   args.local_rank, args.device, args.n_gpu,
                   bool(args.local_rank != -1))
    logger.info(dict(args._get_kwargs()))
    if args.seed is not None:
        set_seed(args)
    setup_dataset_and_model(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')
    labeled_targets = np.asarray(labeled_dataset.targets)
    class_sample_count = reconcile_labeled_class_counts(args, labeled_dataset)
    sample_weights = 1.0 / np.power(
        class_sample_count[labeled_targets], args.minor_balanced_sampler_alpha)
    sample_weights = sample_weights / sample_weights.mean()
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=WeightedRandomSampler(torch.DoubleTensor(sample_weights),
                                      num_samples=len(sample_weights),
                                      replacement=True),
        batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True)
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=torch.utils.data.RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args).to(args.device)
    optimizer = create_optimizer(args, model)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None

    if args.eval_only:
        test_model = ema_model.ema if args.use_ema else model
        _, acc = test(args, test_loader, test_model, 0, 'minor')
        logger.info('Eval single minor top-1: %.2f', acc)
        args.writer.close()
        return

    logger.info('***** Running single minor training *****')
    logger.info('  Num Epochs = %s', args.epochs)
    logger.info('  Total optimization steps = %s', args.total_steps)
    train_single_minor(args, labeled_loader, unlabeled_loader, test_loader,
                       model, optimizer, ema_model, scheduler)


if __name__ == '__main__':
    main()
