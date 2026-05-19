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
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import WeightedRandomSampler
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
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, num_cycles=7. / 16.,
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


def create_optimizer(args, model):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    return optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9,
                     nesterov=args.nesterov)


def linear_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    return float(np.clip(current / rampup_length, 0.0, 1.0))


def minority_class_weights(class_counts, args):
    counts = class_counts.float().clamp_min(1.0)
    weights = torch.pow(counts.max() / counts, args.minority_supervised_gamma)
    weights = torch.clamp(weights, max=args.max_minority_weight)
    return weights / weights.mean().clamp_min(1e-12)


def ramped_class_weights(class_counts, args, ramp):
    target = minority_class_weights(class_counts, args)
    weights = torch.ones_like(target) * (1.0 - ramp) + target * ramp
    return weights / weights.mean().clamp_min(1e-12)


def classwise_topk_mask(max_probs, targets, class_counts, args, min_conf=None):
    mask = torch.zeros_like(max_probs)
    counts = class_counts.to(max_probs.device).float().clamp_min(1.0)
    max_count = counts.max()
    pred_hist = torch.bincount(targets, minlength=args.num_classes).float()
    majority_pred = max(1, int(pred_hist.max().item()))
    majority_quota = max(1, math.ceil(majority_pred * args.major_top_ratio))

    for class_idx in range(args.num_classes):
        class_pos = torch.nonzero(targets.eq(class_idx), as_tuple=False).view(-1)
        if class_pos.numel() == 0:
            continue
        if args.classwise_quota == 'batch':
            base_k = math.ceil(class_pos.numel() * args.major_top_ratio)
        else:
            base_k = majority_quota
        balance = torch.pow(max_count / counts[class_idx],
                            args.minority_threshold_gamma).item()
        k = min(class_pos.numel(), max(1, math.ceil(base_k * balance)))
        keep_rel = torch.topk(max_probs[class_pos], k=k, largest=True).indices
        keep_pos = class_pos[keep_rel]
        confidence_floor = args.min_threshold if min_conf is None else min_conf
        keep_pos = keep_pos[max_probs[keep_pos].ge(confidence_floor)]
        mask[keep_pos] = 1.0
    return mask


def daso_minority_probs(logits, class_counts, pseudo_dist, args, bias_scale=1.0):
    counts = class_counts.to(logits.device).float().clamp_min(1.0)
    prior = counts / counts.sum()
    linear_probs = torch.softmax(logits.detach() / args.T, dim=-1)
    minority_logits = logits.detach() / args.T - \
        bias_scale * args.minority_bias_strength * torch.log(prior.view(1, -1))
    semantic_proxy = torch.softmax(minority_logits, dim=-1)

    dist = pseudo_dist.to(logits.device).float().clamp_min(1e-6)
    rarity = 1.0 - dist / dist.max()
    blend = bias_scale * torch.clamp(
        args.daso_blend_base + rarity * args.daso_blend_scale,
        min=0.0, max=1.0)
    probs = (1.0 - blend.view(1, -1)) * linear_probs + \
        blend.view(1, -1) * semantic_proxy
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def update_pseudo_dist(pseudo_dist, targets, args):
    hist = torch.bincount(targets, minlength=args.num_classes).float()
    hist = hist / hist.sum().clamp_min(1.0)
    pseudo_dist.mul_(args.pseudo_dist_momentum).add_(
        hist.cpu(), alpha=1.0 - args.pseudo_dist_momentum)


def build_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
    else:
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    logger.info("Total params: %.2fM",
                sum(p.numel() for p in model.parameters()) / 1e6)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
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
    parser.add_argument('--dual-bias-cotrain', action='store_true')
    parser.add_argument('--imb-ratio', default=1.0, type=float)
    parser.add_argument('--imb-type', default='exp', choices=['exp', 'step'])
    parser.add_argument('--major-top-ratio', default=0.2, type=float)
    parser.add_argument('--classwise-quota', default='global',
                        choices=['global', 'batch'])
    parser.add_argument('--minority-threshold-gamma', default=1.0, type=float)
    parser.add_argument('--min-threshold', default=0.5, type=float)
    parser.add_argument('--minority-bias-strength', default=1.0, type=float)
    parser.add_argument('--minority-supervised-gamma', default=1.0, type=float)
    parser.add_argument('--minor-balanced-labeled', action='store_true',
                        default=True)
    parser.add_argument('--no-minor-balanced-labeled', action='store_true')
    parser.add_argument('--max-minority-weight', default=10.0, type=float)
    parser.add_argument('--minority-bias-warmup', default=2048, type=int)
    parser.add_argument('--daso-blend-base', default=0.2, type=float)
    parser.add_argument('--daso-blend-scale', default=0.6, type=float)
    parser.add_argument('--pseudo-dist-momentum', default=0.999, type=float)
    parser.add_argument('--agreement-threshold', default=0.5, type=float)
    parser.add_argument('--consistency-weight', default=0.1, type=float)
    parser.add_argument('--ensemble-minor-weight', default=0.7, type=float)
    parser.add_argument('--dual-final-predictor', default='ensemble',
                        choices=['major', 'minor', 'ensemble', 'conf',
                                 'major_adjusted', 'minor_gate'])
    parser.add_argument('--final-logit-adjust-tau', default=1.6, type=float)
    parser.add_argument('--selection-metric', default='final',
                        choices=['final', 'ensemble', 'best'])
    parser.add_argument('--unsup-warmup', default=1024, type=int)
    parser.add_argument('--pseudo-warmup', default=512, type=int)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--init-checkpoint', default='', type=str)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    if args.no_ema:
        args.use_ema = False
    if args.no_minor_balanced_labeled:
        args.minor_balanced_labeled = False
    return args


def main():
    global best_acc
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
    logger.warning('Process rank: %s, device: %s, n_gpu: %s, '
                   'distributed training: %s, 16-bits training: %s',
                   args.local_rank, args.device, args.n_gpu,
                   bool(args.local_rank != -1), args.amp)
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

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

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset, sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True)

    minor_labeled_trainloader = None
    if args.dual_bias_cotrain and args.minor_balanced_labeled:
        labeled_targets = np.array(labeled_dataset.targets)
        class_sample_count = np.bincount(
            labeled_targets, minlength=args.num_classes).astype(np.float32)
        sample_weights = 1.0 / np.maximum(class_sample_count[labeled_targets], 1.0)
        minor_labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=WeightedRandomSampler(torch.DoubleTensor(sample_weights),
                                          num_samples=len(sample_weights),
                                          replacement=True),
            batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset, sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args).to(args.device)
    model_minor = build_model(args).to(args.device) if args.dual_bias_cotrain else None
    optimizer = create_optimizer(args, model)
    optimizer_minor = create_optimizer(args, model_minor) if args.dual_bias_cotrain else None
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    scheduler_minor = get_cosine_schedule_with_warmup(
        optimizer_minor, args.warmup, args.total_steps) if args.dual_bias_cotrain else None

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
        ema_model_minor = ModelEMA(args, model_minor, args.ema_decay) \
            if args.dual_bias_cotrain else None
    else:
        ema_model = None
        ema_model_minor = None

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        init_state = checkpoint.get('ema_state_dict', checkpoint.get('state_dict'))
        model.load_state_dict(init_state)
        if model_minor is not None:
            model_minor.load_state_dict(init_state)
        if args.use_ema:
            ema_model.ema.load_state_dict(init_state)
            if ema_model_minor is not None:
                ema_model_minor.ema.load_state_dict(init_state)

    if args.resume:
        logger.info('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location='cpu')
        best_acc = checkpoint.get('best_acc', checkpoint.get('acc', 0))
        args.start_epoch = checkpoint.get('epoch', 0)
        model.load_state_dict(checkpoint.get('state_dict_major',
                                             checkpoint.get('state_dict')))
        if model_minor is not None:
            model_minor.load_state_dict(checkpoint.get('state_dict_minor',
                                                       checkpoint.get('state_dict')))
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint.get(
                'ema_state_dict_major', checkpoint.get('ema_state_dict')))
            if ema_model_minor is not None:
                ema_model_minor.ema.load_state_dict(checkpoint.get(
                    'ema_state_dict_minor', checkpoint.get('ema_state_dict')))
        if not args.eval_only:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if optimizer_minor is not None and 'optimizer_minor' in checkpoint:
                optimizer_minor.load_state_dict(checkpoint['optimizer_minor'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if scheduler_minor is not None and 'scheduler_minor' in checkpoint:
                scheduler_minor.load_state_dict(checkpoint['scheduler_minor'])

    if args.eval_only:
        if args.dual_bias_cotrain:
            major = ema_model.ema if args.use_ema else model
            minor = ema_model_minor.ema if args.use_ema else model_minor
            _, acc_major = test(args, test_loader, major, 0)
            _, acc_minor = test(args, test_loader, minor, 0)
            _, acc_ens = test_ensemble(args, test_loader, major, minor)
            if args.dual_final_predictor == 'major':
                acc_final = acc_major
            elif args.dual_final_predictor == 'minor':
                acc_final = acc_minor
            elif args.dual_final_predictor == 'ensemble':
                acc_final = acc_ens
            else:
                _, acc_final = test_dual_final(args, test_loader, major, minor)
            logger.info('Eval major/minor/ensemble top-1: %.2f / %.2f / %.2f',
                        acc_major, acc_minor, acc_ens)
            logger.info('Eval final top-1: %.2f', acc_final)
        else:
            test(args, test_loader, ema_model.ema if args.use_ema else model, 0)
        args.writer.close()
        return

    logger.info('***** Running training *****')
    logger.info('  Task = %s@%s', args.dataset, args.num_labeled)
    logger.info('  Num Epochs = %s', args.epochs)
    logger.info('  Batch size per GPU = %s', args.batch_size)
    logger.info('  Total train batch size = %s', args.batch_size * args.world_size)
    logger.info('  Total optimization steps = %s', args.total_steps)

    if args.dual_bias_cotrain:
        train_dual_bias(args, labeled_trainloader, minor_labeled_trainloader,
                        unlabeled_trainloader, test_loader, model, model_minor,
                        optimizer, optimizer_minor, ema_model, ema_model_minor,
                        scheduler, scheduler_minor)
    else:
        train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
              model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    end = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        p_bar = tqdm(range(args.eval_step), disable=args.no_progress)
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                                2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = de_interleave(model(inputs), 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
            loss = Lx + args.lambda_u * Lu
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())
            batch_time.update(time.time() - end)
            end = time.time()
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
        test_loss, test_acc = test(args, test_loader, test_model, epoch)
        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
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
        logger.info('Best top-1 acc: %.2f', best_acc)
        logger.info('Mean top-1 acc: %.2f\n', np.mean(test_accs[-20:]))
    args.writer.close()


def train_dual_bias(args, labeled_trainloader, minor_labeled_trainloader,
                    unlabeled_trainloader, test_loader, model_major, model_minor,
                    optimizer_major, optimizer_minor, ema_major, ema_minor,
                    scheduler_major, scheduler_minor):
    global best_acc
    test_accs = []
    labeled_iter = iter(labeled_trainloader)
    minor_labeled_iter = iter(minor_labeled_trainloader) \
        if minor_labeled_trainloader is not None else None
    unlabeled_iter = iter(unlabeled_trainloader)
    class_counts = torch.tensor(args.labeled_class_counts, dtype=torch.float)
    pseudo_dist = torch.ones(args.num_classes) / args.num_classes
    global_step = args.start_epoch * args.eval_step
    end = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        model_major.train()
        model_minor.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_major_probs = AverageMeter()
        mask_minor_probs = AverageMeter()
        p_bar = tqdm(range(args.eval_step), disable=args.no_progress)
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            if minor_labeled_iter is not None:
                try:
                    inputs_x_minor, targets_x_minor = next(minor_labeled_iter)
                except StopIteration:
                    minor_labeled_iter = iter(minor_labeled_trainloader)
                    inputs_x_minor, targets_x_minor = next(minor_labeled_iter)
            else:
                inputs_x_minor, targets_x_minor = inputs_x, targets_x

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]
            inputs_major = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                2 * args.mu + 1).to(args.device)
            inputs_minor = interleave(
                torch.cat((inputs_x_minor, inputs_u_w, inputs_u_s)),
                2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_x_minor = targets_x_minor.to(args.device)

            logits_major = de_interleave(model_major(inputs_major),
                                         2 * args.mu + 1)
            logits_minor = de_interleave(model_minor(inputs_minor),
                                         2 * args.mu + 1)
            logits_x_major = logits_major[:batch_size]
            logits_u_w_major, logits_u_s_major = logits_major[batch_size:].chunk(2)
            logits_x_minor = logits_minor[:batch_size]
            logits_u_w_minor, logits_u_s_minor = logits_minor[batch_size:].chunk(2)

            bias_ramp = linear_rampup(global_step, args.minority_bias_warmup)
            minor_sup_weight = ramped_class_weights(
                class_counts, args, bias_ramp).to(args.device)
            Lx_major = F.cross_entropy(logits_x_major, targets_x, reduction='mean')
            Lx_minor = F.cross_entropy(logits_x_minor, targets_x_minor,
                                       weight=minor_sup_weight, reduction='mean')

            probs_major = torch.softmax(logits_u_w_major.detach() / args.T, dim=-1)
            max_major, targets_major = torch.max(probs_major, dim=-1)
            probs_minor = daso_minority_probs(
                logits_u_w_minor, class_counts, pseudo_dist, args, bias_ramp)
            max_minor, targets_minor = torch.max(probs_minor, dim=-1)

            threshold_ramp = linear_rampup(global_step - args.pseudo_warmup,
                                           args.minority_bias_warmup)
            confidence_floor = args.threshold - \
                (args.threshold - args.min_threshold) * threshold_ramp
            mask_major = classwise_topk_mask(
                max_major, targets_major, class_counts, args, confidence_floor)
            mask_minor = classwise_topk_mask(
                max_minor, targets_minor, class_counts, args, confidence_floor)

            agree = targets_major.eq(targets_minor)
            agree_conf = torch.minimum(max_major, max_minor).ge(
                args.agreement_threshold)
            mask_union = torch.maximum(mask_major, mask_minor)
            mask_major = torch.where(agree & agree_conf, mask_union, mask_major)
            mask_minor = torch.where(agree & agree_conf, mask_union, mask_minor)
            if global_step < args.pseudo_warmup:
                mask_major = torch.zeros_like(mask_major)
                mask_minor = torch.zeros_like(mask_minor)

            update_pseudo_dist(pseudo_dist, targets_minor.cpu(), args)
            Lu_major = (F.cross_entropy(logits_u_s_major, targets_minor,
                                        reduction='none') * mask_minor).mean()
            Lu_minor = (F.cross_entropy(logits_u_s_minor, targets_major,
                                        reduction='none') * mask_major).mean()

            prob_s_major = torch.softmax(logits_u_s_major, dim=-1)
            prob_s_minor = torch.softmax(logits_u_s_minor, dim=-1)
            consistency = 0.5 * (
                F.kl_div(torch.log(prob_s_major.clamp_min(1e-12)),
                         prob_s_minor.detach(), reduction='batchmean') +
                F.kl_div(torch.log(prob_s_minor.clamp_min(1e-12)),
                         prob_s_major.detach(), reduction='batchmean'))

            Lx = Lx_major + Lx_minor
            Lu = Lu_major + Lu_minor
            ramp = linear_rampup(global_step - args.pseudo_warmup,
                                 args.unsup_warmup)
            loss = Lx + ramp * args.lambda_u * Lu + \
                ramp * args.consistency_weight * consistency
            loss.backward()
            optimizer_major.step()
            optimizer_minor.step()
            scheduler_major.step()
            scheduler_minor.step()
            if args.use_ema:
                ema_major.update(model_major)
                ema_minor.update(model_minor)
            model_major.zero_grad()
            model_minor.zero_grad()
            global_step += 1

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_major_probs.update(mask_major.mean().item())
            mask_minor_probs.update(mask_minor.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    'Train Epoch: {}/{}. Iter: {}/{}. LR: {:.4f}. Loss: {:.4f}. '
                    'Loss_x: {:.4f}. Loss_u: {:.4f}. MaskMajor: {:.2f}. '
                    'MaskMinor: {:.2f}.'.format(
                        epoch + 1, args.epochs, batch_idx + 1, args.eval_step,
                        scheduler_major.get_last_lr()[0], losses.avg,
                        losses_x.avg, losses_u.avg, mask_major_probs.avg,
                        mask_minor_probs.avg))
                p_bar.update()
            end = time.time()

        if not args.no_progress:
            p_bar.close()

        test_model_major = ema_major.ema if args.use_ema else model_major
        test_model_minor = ema_minor.ema if args.use_ema else model_minor
        test_loss_major, test_acc_major = test(
            args, test_loader, test_model_major, epoch)
        test_loss_minor, test_acc_minor = test(
            args, test_loader, test_model_minor, epoch)
        test_loss_ens, test_acc_ens = test_ensemble(
            args, test_loader, test_model_major, test_model_minor)
        if args.dual_final_predictor == 'major':
            test_loss_final, test_acc_final = test_loss_major, test_acc_major
        elif args.dual_final_predictor == 'minor':
            test_loss_final, test_acc_final = test_loss_minor, test_acc_minor
        elif args.dual_final_predictor == 'ensemble':
            test_loss_final, test_acc_final = test_loss_ens, test_acc_ens
        else:
            test_loss_final, test_acc_final = test_dual_final(
                args, test_loader, test_model_major, test_model_minor)

        if args.selection_metric == 'best':
            candidates = [(test_acc_major, test_loss_major),
                          (test_acc_minor, test_loss_minor),
                          (test_acc_ens, test_loss_ens),
                          (test_acc_final, test_loss_final)]
            test_acc, test_loss = max(candidates, key=lambda item: item[0])
        elif args.selection_metric == 'ensemble':
            test_acc, test_loss = test_acc_ens, test_loss_ens
        else:
            test_acc, test_loss = test_acc_final, test_loss_final

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/4.mask_major', mask_major_probs.avg, epoch)
        args.writer.add_scalar('train/5.mask_minor', mask_minor_probs.avg, epoch)
        args.writer.add_scalar('test/1.test_acc_final', test_acc_final, epoch)
        args.writer.add_scalar('test/2.test_acc_major', test_acc_major, epoch)
        args.writer.add_scalar('test/3.test_acc_minor', test_acc_minor, epoch)
        args.writer.add_scalar('test/4.test_acc_ensemble', test_acc_ens, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        major_to_save = model_major.module if hasattr(model_major, 'module') else model_major
        minor_to_save = model_minor.module if hasattr(model_minor, 'module') else model_minor
        ema_major_to_save = ema_major.ema.module if args.use_ema and hasattr(
            ema_major.ema, 'module') else (ema_major.ema if args.use_ema else None)
        ema_minor_to_save = ema_minor.ema.module if args.use_ema and hasattr(
            ema_minor.ema, 'module') else (ema_minor.ema if args.use_ema else None)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': major_to_save.state_dict(),
            'state_dict_major': major_to_save.state_dict(),
            'state_dict_minor': minor_to_save.state_dict(),
            'ema_state_dict': ema_major_to_save.state_dict() if args.use_ema else None,
            'ema_state_dict_major': ema_major_to_save.state_dict() if args.use_ema else None,
            'ema_state_dict_minor': ema_minor_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'acc_major': test_acc_major,
            'acc_minor': test_acc_minor,
            'acc_ensemble': test_acc_ens,
            'acc_final': test_acc_final,
            'best_acc': best_acc,
            'optimizer': optimizer_major.state_dict(),
            'optimizer_minor': optimizer_minor.state_dict(),
            'scheduler': scheduler_major.state_dict(),
            'scheduler_minor': scheduler_minor.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Major top-1 acc: %.2f', test_acc_major)
        logger.info('Minor top-1 acc: %.2f', test_acc_minor)
        logger.info('Ensemble top-1 acc: %.2f', test_acc_ens)
        logger.info('Final top-1 acc: %.2f', test_acc_final)
        logger.info('Best final top-1 acc: %.2f', best_acc)
        logger.info('Mean final top-1 acc: %.2f\n', np.mean(test_accs[-20:]))
    args.writer.close()


def test(args, test_loader, model, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
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
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
    logger.info('top-1 acc: %.2f', top1.avg)
    logger.info('top-5 acc: %.2f', top5.avg)
    per_class = class_correct / class_total.clamp_min(1.0) * 100.0
    logger.info('per-class acc: %s', [round(v, 2) for v in per_class.tolist()])
    log_groups(args, per_class, '')
    return losses.avg, top1.avg


def test_ensemble(args, test_loader, model_major, model_minor):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
    model_major.eval()
    model_minor.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_major = model_major(inputs)
            outputs_minor = model_minor(inputs)
            minor_w = args.ensemble_minor_weight
            outputs = (1.0 - minor_w) * outputs_major + minor_w * outputs_minor
            loss = F.cross_entropy(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            for class_idx in range(args.num_classes):
                class_mask = targets.eq(class_idx)
                class_total[class_idx] += class_mask.sum().cpu()
                class_correct[class_idx] += preds[class_mask].eq(
                    targets[class_mask]).sum().cpu()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
    logger.info('ensemble top-1 acc: %.2f', top1.avg)
    logger.info('ensemble top-5 acc: %.2f', top5.avg)
    per_class = class_correct / class_total.clamp_min(1.0) * 100.0
    logger.info('ensemble per-class acc: %s',
                [round(v, 2) for v in per_class.tolist()])
    log_groups(args, per_class, 'ensemble ')
    return losses.avg, top1.avg


def test_dual_final(args, test_loader, model_major, model_minor):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
    model_major.eval()
    model_minor.eval()
    counts = None
    log_prior = None
    if hasattr(args, 'labeled_class_counts'):
        counts = torch.tensor(args.labeled_class_counts, dtype=torch.float,
                              device=args.device).clamp_min(1.0)
        log_prior = torch.log((counts / counts.sum()).view(1, -1))
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_major = model_major(inputs)
            outputs_minor = model_minor(inputs)
            if args.dual_final_predictor == 'major_adjusted':
                outputs = outputs_major - args.final_logit_adjust_tau * log_prior
            elif args.dual_final_predictor == 'minor_gate':
                pred_minor = outputs_minor.argmax(dim=1, keepdim=True)
                counts_cpu = torch.tensor(args.labeled_class_counts,
                                          dtype=torch.float)
                sorted_counts = torch.sort(counts_cpu, descending=True).values
                threshold = sorted_counts[max(0, args.num_classes // 2 - 1)]
                tail_mask = counts[pred_minor].lt(threshold).to(outputs_major.device)
                outputs = torch.where(tail_mask, outputs_minor, outputs_major)
            else:
                probs_major = torch.softmax(outputs_major, dim=1)
                probs_minor = torch.softmax(outputs_minor, dim=1)
                conf_major, _ = torch.max(probs_major, dim=1, keepdim=True)
                conf_minor, _ = torch.max(probs_minor, dim=1, keepdim=True)
                outputs = torch.where(conf_minor.ge(conf_major),
                                      outputs_minor, outputs_major)
            loss = F.cross_entropy(outputs, targets)
            preds = torch.argmax(outputs, dim=1)
            for class_idx in range(args.num_classes):
                class_mask = targets.eq(class_idx)
                class_total[class_idx] += class_mask.sum().cpu()
                class_correct[class_idx] += preds[class_mask].eq(
                    targets[class_mask]).sum().cpu()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
    logger.info('%s final top-1 acc: %.2f', args.dual_final_predictor, top1.avg)
    logger.info('%s final top-5 acc: %.2f', args.dual_final_predictor, top5.avg)
    per_class = class_correct / class_total.clamp_min(1.0) * 100.0
    logger.info('%s final per-class acc: %s', args.dual_final_predictor,
                [round(v, 2) for v in per_class.tolist()])
    log_groups(args, per_class, args.dual_final_predictor + ' final ')
    return losses.avg, top1.avg


def log_groups(args, per_class, prefix):
    if not hasattr(args, 'labeled_class_counts'):
        return
    sorted_cls = sorted(range(args.num_classes),
                        key=lambda i: args.labeled_class_counts[i],
                        reverse=True)
    group = max(1, args.num_classes // 3)
    head = sorted_cls[:group]
    mid = sorted_cls[group:2 * group]
    tail = sorted_cls[2 * group:]
    logger.info('%shead/mid/tail acc: %.2f / %.2f / %.2f', prefix,
                per_class[head].mean().item(),
                per_class[mid].mean().item(),
                per_class[tail].mean().item())


if __name__ == '__main__':
    main()
