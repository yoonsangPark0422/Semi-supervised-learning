import argparse
import os
import subprocess
import sys
from datetime import datetime


def timestamp():
    return datetime.now().isoformat(timespec='seconds')


def run_one(python, project_dir, common_args, result_root, seed, name, extra):
    out_dir = os.path.join(result_root, f'seed_{seed}', name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train.log')
    args = common_args + ['--seed', str(seed), '--out', out_dir] + extra
    checkpoint = os.path.join(out_dir, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint):
        args += ['--resume', checkpoint]

    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'[{timestamp()}] Starting seed={seed} {name}\n')
        log.flush()
        process = subprocess.run(
            [python] + args,
            stdout=log,
            stderr=log,
            cwd=project_dir,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f'Training failed for {name} seed={seed} '
                f'with exit code {process.returncode}'
            )
        log.write(f'[{timestamp()}] Finished seed={seed} {name}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', default='results/epoch200_compare')
    parser.add_argument('--seeds', default='5')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-step', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mu', type=int, default=7)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.result_root):
        args.result_root = os.path.join(project_dir, args.result_root)

    python = sys.executable
    total_steps = args.epochs * args.eval_step
    common_args = [
        'train.py',
        '--dataset', 'cifar10',
        '--num-labeled', '4000',
        '--arch', 'wideresnet',
        '--batch-size', str(args.batch_size),
        '--mu', str(args.mu),
        '--lr', '0.03',
        '--expand-labels',
        '--imb-ratio', '100',
        '--total-steps', str(total_steps),
        '--eval-step', str(args.eval_step),
        '--no-progress',
    ]

    experiments = [
        ('00_fixmatch', ['--ablation-preset', 'fixmatch']),
        ('01_dual_only', [
            '--ablation-preset', 'full',
            '--no-minor-balanced-labeled',
            '--no-minor-class-weight',
            '--no-minority-pseudo-bias',
            '--no-classwise-pseudo-mask',
            '--no-dual-agreement',
            '--no-dual-consistency',
        ]),
        ('02_best_sampler_class_weight_semantic_proxy_topk', [
            '--ablation-preset', 'full',
            '--no-dual-agreement',
            '--no-dual-consistency',
        ]),
        ('03_full_dual_with_agreement_consistency', [
            '--ablation-preset', 'full',
        ]),
    ]

    seeds = [int(seed.strip()) for seed in args.seeds.split(',') if seed.strip()]
    os.makedirs(args.result_root, exist_ok=True)
    runner_log = os.path.join(args.result_root, 'runner.log')
    with open(runner_log, 'a', encoding='utf-8') as log:
        log.write(
            f'[{timestamp()}] epochs={args.epochs} eval_step={args.eval_step} '
            f'total_steps={total_steps} batch_size={args.batch_size} '
            f'mu={args.mu} seeds={seeds}\n'
        )

    for seed in seeds:
        for name, extra in experiments:
            run_one(python, project_dir, common_args, args.result_root,
                    seed, name, extra)

    subprocess.run(
        [python, 'summarize_ablation.py', '--root', args.result_root],
        check=True,
        cwd=project_dir,
    )


if __name__ == '__main__':
    main()
