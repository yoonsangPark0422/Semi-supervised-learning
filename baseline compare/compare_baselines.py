import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DUAL_DIR = ROOT
sys.path.insert(0, str(DUAL_DIR))

from dataset.cifar import DATASET_GETTERS  # noqa: E402
from train import (  # noqa: E402
    AverageMeter,
    PseudoLabelStats,
    accuracy,
    build_model,
    create_optimizer,
    daso_minority_probs,
    de_interleave,
    get_cosine_schedule_with_warmup,
    interleave,
    log_groups,
    reset_peak_memory,
    get_peak_memory,
    set_seed,
    write_pseudo_stats,
    write_train_metrics,
)


LOGGER = logging.getLogger(__name__)


def append_csv(path, fields, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_args(cli, method, out_dir):
    args = SimpleNamespace()
    args.gpu_id = cli.gpu_id
    args.num_workers = cli.num_workers
    args.dataset = 'cifar10'
    args.num_labeled = cli.num_labeled
    args.expand_labels = True
    args.arch = 'wideresnet'
    args.total_steps = cli.epochs * cli.eval_step
    args.eval_step = cli.eval_step
    args.start_epoch = 0
    args.batch_size = cli.batch_size
    args.lr = cli.lr
    args.warmup = 0
    args.wdecay = 5e-4
    args.nesterov = True
    args.use_ema = True
    args.no_ema = False
    args.ema_decay = 0.999
    args.mu = cli.mu
    args.lambda_u = 1
    args.T = 1
    args.threshold = cli.threshold
    args.imb_ratio = cli.imb_ratio
    args.imb_type = 'exp'
    args.major_top_ratio = 0.2
    args.classwise_quota = 'global'
    args.minority_threshold_gamma = 1.0
    args.min_threshold = 0.5
    args.minority_bias_strength = 1.0
    args.minority_supervised_gamma = 1.0
    args.minor_balanced_labeled = True
    args.no_minor_balanced_labeled = False
    args.no_minor_class_weight = False
    args.no_minority_pseudo_bias = False
    args.no_classwise_pseudo_mask = False
    args.no_dual_agreement = False
    args.no_dual_consistency = False
    args.max_minority_weight = 10.0
    args.minority_bias_warmup = cli.eval_step * 2
    args.daso_blend_base = 0.2
    args.daso_blend_scale = 0.6
    args.pseudo_dist_momentum = 0.999
    args.agreement_threshold = 0.5
    args.consistency_weight = 0.1
    args.ensemble_minor_weight = 0.7
    args.dual_final_predictor = 'ensemble'
    args.final_logit_adjust_tau = 1.6
    args.selection_metric = 'final'
    args.export_features = False
    args.feature_export_max = 2000
    args.unsup_warmup = cli.eval_step
    args.pseudo_warmup = max(1, cli.eval_step // 2)
    args.out = str(out_dir)
    args.resume = ''
    args.init_checkpoint = ''
    args.eval_only = False
    args.no_save_checkpoint = False
    args.seed = cli.seed
    args.amp = False
    args.opt_level = 'O1'
    args.local_rank = -1
    args.no_progress = True
    args.method = method
    args.num_classes = 10
    args.model_depth = 28
    args.model_width = 2
    args.model_cardinality = 4
    args.device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() \
        else torch.device('cpu')
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.epochs = cli.epochs
    return args


def make_loaders(args):
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, str(DUAL_DIR / 'data'))
    labeled_loader = DataLoader(
        labeled_dataset, sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers)

    labels = np.array(labeled_dataset.targets)
    counts = np.bincount(labels, minlength=args.num_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    balanced_loader = DataLoader(
        labeled_dataset,
        sampler=WeightedRandomSampler(torch.DoubleTensor(weights),
                                      num_samples=len(weights),
                                      replacement=True),
        batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True)
    return labeled_loader, balanced_loader, unlabeled_loader, test_loader


def test_model(args, loader, model, epoch, name='model'):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_correct = torch.zeros(args.num_classes)
    class_total = torch.zeros(args.num_classes)
    confusion = torch.zeros(args.num_classes, args.num_classes)
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            preds = outputs.argmax(dim=1)
            for class_idx in range(args.num_classes):
                class_mask = targets.eq(class_idx)
                class_total[class_idx] += class_mask.sum().cpu()
                class_correct[class_idx] += preds[class_mask].eq(
                    targets[class_mask]).sum().cpu()
            for true_class, pred_class in zip(targets.cpu(), preds.cpu()):
                confusion[true_class.long(), pred_class.long()] += 1
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

    per_class = class_correct / class_total.clamp_min(1.0) * 100.0
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    macro_f1 = f1.mean().item() * 100.0

    fields = ['epoch', 'model', 'class', 'accuracy', 'f1', 'support']
    for class_idx in range(args.num_classes):
        append_csv(os.path.join(args.out, 'per_class_eval.csv'), fields, {
            'epoch': epoch + 1,
            'model': name,
            'class': class_idx,
            'accuracy': per_class[class_idx].item(),
            'f1': f1[class_idx].item() * 100.0,
            'support': class_total[class_idx].item(),
        })
    log_groups(args, per_class, name + ' ')
    return losses.avg, top1.avg, top5.avg, macro_f1


class FeatureHook(object):
    def __init__(self, model):
        self.features = None
        classifier = model.fc if hasattr(model, 'fc') else model.classifier
        self.handle = classifier.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.features = inputs[0]

    def close(self):
        self.handle.remove()


def crest_weights(args, targets, epoch):
    counts = torch.tensor(args.labeled_class_counts, dtype=torch.float,
                          device=targets.device).clamp_min(1.0)
    rarity = (counts.max() / counts).pow(0.5)
    ramp = min(1.0, float(epoch + 1) / max(1, args.epochs // 2))
    weights = 1.0 + ramp * (rarity[targets] - 1.0)
    return weights / weights.mean().clamp_min(1e-12)


def train_baseline(args):
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)
    set_seed(args)
    labeled_loader, balanced_loader, unlabeled_loader, test_loader = make_loaders(args)

    model = build_model(args).to(args.device)
    optimizer = create_optimizer(args, model)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    from models.ema import ModelEMA
    ema_model = ModelEMA(args, model, args.ema_decay)

    aux_head = None
    hook = None
    aux_optimizer = None
    if args.method == 'abc':
        aux_head = nn.Linear(model.channels, args.num_classes).to(args.device)
        hook = FeatureHook(model)
        aux_optimizer = torch.optim.SGD(aux_head.parameters(), lr=args.lr,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=args.wdecay)

    labeled_iter = iter(labeled_loader)
    balanced_iter = iter(balanced_loader)
    unlabeled_iter = iter(unlabeled_loader)
    class_counts = torch.tensor(args.labeled_class_counts, dtype=torch.float)
    pseudo_dist = torch.ones(args.num_classes) / args.num_classes
    global_step = 0
    best_acc = 0.0
    start_epoch = 0
    checkpoint_path = os.path.join(args.out, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if aux_head is not None and checkpoint.get('aux_head') is not None:
            aux_head.load_state_dict(checkpoint['aux_head'])
        if aux_optimizer is not None and checkpoint.get('aux_optimizer') is not None:
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        if checkpoint.get('pseudo_dist') is not None:
            pseudo_dist = checkpoint['pseudo_dist'].cpu()
        start_epoch = int(checkpoint.get('epoch', 0))
        best_acc = float(checkpoint.get('best_acc', 0.0))
        global_step = start_epoch * args.eval_step
        LOGGER.info('Resumed %s from epoch %s/%s best %.2f',
                    args.method, start_epoch, args.epochs, best_acc)

    for epoch in range(start_epoch, args.epochs):
        reset_peak_memory(args)
        epoch_start = time.time()
        model.train()
        if aux_head is not None:
            aux_head.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        pseudo_stats = PseudoLabelStats(args.num_classes)
        iterator = tqdm(range(args.eval_step), disable=args.no_progress)
        for _ in iterator:
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)
            try:
                inputs_b, targets_b = next(balanced_iter)
            except StopIteration:
                balanced_iter = iter(balanced_loader)
                inputs_b, targets_b = next(balanced_iter)
            try:
                (inputs_u_w, inputs_u_s), targets_u_true = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s), targets_u_true = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                                2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_b = targets_b.to(args.device)
            targets_u_true = targets_u_true.to(args.device)
            logits = de_interleave(model(inputs), 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            if args.method == 'daso':
                probs = daso_minority_probs(
                    logits_u_w, class_counts, pseudo_dist, args,
                    min(1.0, global_step / max(1, args.minority_bias_warmup)))
            else:
                probs = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(probs, dim=-1)
            threshold = args.threshold
            if args.method == 'crest':
                counts = class_counts.to(args.device).clamp_min(1.0)
                threshold = args.threshold - 0.2 * (
                    1.0 - counts[targets_u] / counts.max())
            mask = max_probs.ge(threshold).float()
            pseudo_stats.update(targets_u, targets_u_true, mask)

            Lu_item = F.cross_entropy(logits_u_s, targets_u, reduction='none')
            if args.method == 'crest':
                Lu_item = Lu_item * crest_weights(args, targets_u, epoch)
            Lu = (Lu_item * mask).mean()
            loss = Lx + args.lambda_u * Lu

            if args.method == 'abc':
                logits_b = model(inputs_b.to(args.device))
                features_b = hook.features
                logits_b_aux = aux_head(features_b)
                aux_counts = class_counts.to(args.device).clamp_min(1.0)
                aux_weight = (aux_counts.max() / aux_counts).clamp(max=10.0)
                aux_weight = aux_weight / aux_weight.mean()
                loss = loss + F.cross_entropy(
                    logits_b_aux, targets_b, weight=aux_weight)
            elif args.method == 'daso':
                update = torch.bincount(
                    targets_u.detach().cpu(), minlength=args.num_classes).float()
                update = update / update.sum().clamp_min(1.0)
                pseudo_dist.mul_(args.pseudo_dist_momentum).add_(
                    update, alpha=1.0 - args.pseudo_dist_momentum)

            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if aux_optimizer is not None:
                aux_optimizer.step()
            scheduler.step()
            ema_model.update(model)
            global_step += 1

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())

        test_model = ema_model.ema
        test_loss, test_acc, test_top5, macro_f1 = test_model_eval(
            args, test_loader, test_model, epoch)
        peak_allocated, peak_reserved = get_peak_memory(args)
        write_pseudo_stats(args, epoch, args.method, pseudo_stats)
        write_train_metrics(args, epoch, {
            'mode': args.method,
            'epoch_time_sec': time.time() - epoch_start,
            'peak_allocated_mb': peak_allocated,
            'peak_reserved_mb': peak_reserved,
            'param_count_m': parameter_count(model) +
            (parameter_count(aux_head) if aux_head is not None else 0.0),
            'train_loss': losses.avg,
            'train_loss_x': losses_x.avg,
            'train_loss_u': losses_u.avg,
            'mask': mask_probs.avg,
            'test_acc': test_acc,
            'macro_f1': macro_f1,
        })
        args.writer.add_scalar('test/acc', test_acc, epoch)
        args.writer.add_scalar('test/macro_f1', macro_f1, epoch)
        best_acc = max(best_acc, test_acc)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.ema.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'aux_head': aux_head.state_dict() if aux_head is not None else None,
            'aux_optimizer': (
                aux_optimizer.state_dict() if aux_optimizer is not None else None),
            'pseudo_dist': pseudo_dist,
        }, os.path.join(args.out, 'checkpoint.pth.tar'))
        LOGGER.info('%s epoch %s/%s acc %.2f macro_f1 %.2f best %.2f',
                    args.method, epoch + 1, args.epochs, test_acc,
                    macro_f1, best_acc)
    if hook is not None:
        hook.close()
    args.writer.close()
    return best_acc


def test_model_eval(args, test_loader, model, epoch):
    return test_model(args, test_loader, model, epoch, args.method)


def run_proposed(cli, out_dir):
    cmd = [
        sys.executable, str(DUAL_DIR / 'train.py'),
        '--dataset', 'cifar10',
        '--num-labeled', str(cli.num_labeled),
        '--arch', 'wideresnet',
        '--batch-size', str(cli.batch_size),
        '--mu', str(cli.mu),
        '--lr', str(cli.lr),
        '--expand-labels',
        '--imb-ratio', str(cli.imb_ratio),
        '--total-steps', str(cli.epochs * cli.eval_step),
        '--eval-step', str(cli.eval_step),
        '--seed', str(cli.seed),
        '--ablation-preset', 'full',
        '--out', str(out_dir),
        '--no-progress',
    ]
    checkpoint = out_dir / 'checkpoint.pth.tar'
    if checkpoint.exists():
        cmd += ['--resume', str(checkpoint)]
    with open(out_dir / 'train.log', 'a', encoding='utf-8') as log:
        process = subprocess.run(cmd, cwd=str(DUAL_DIR), stdout=log, stderr=log)
    if process.returncode != 0:
        raise RuntimeError('proposed full dual failed')


def summarize(root):
    script = DUAL_DIR / 'summarize_ablation.py'
    subprocess.run([sys.executable, str(script), '--root', str(root)],
                   check=True, cwd=str(DUAL_DIR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default='crest,daso,abc,proposed')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mu', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--imb-ratio', type=float, default=100.0)
    parser.add_argument('--num-labeled', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--out', default=str(ROOT / 'result'))
    return parser.parse_args()


def main():
    cli = parse_args()
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    root = Path(cli.out)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in cli.methods.split(',') if method.strip()]
    for method in methods:
        out_dir = root / f'seed_{cli.seed}' / method
        out_dir.mkdir(parents=True, exist_ok=True)
        if method == 'proposed':
            run_proposed(cli, out_dir)
        else:
            args = make_args(cli, method, out_dir)
            train_baseline(args)
    summarize(root)


if __name__ == '__main__':
    main()
