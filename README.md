# Major-Minor Dual FixMatch

This repository contains the cleaned dual-model training code used for the
imbalanced semi-supervised learning experiments.

## What This Code Does

The dual version trains two FixMatch models at the same time:

- `major` model: standard FixMatch-style training focused on high-confidence
  pseudo-labels.
- `minor` model: minority-aware FixMatch training that can use balanced labeled
  sampling and class-weighted supervised CE.
- Cross pseudo-supervision: each model also learns from the other model's
  pseudo-labels on unlabeled data.
- Minority-biased pseudo-label probabilities: the minor model adjusts pseudo
  probabilities using labeled class counts.
- Class-wise pseudo-label selection: pseudo-labels are selected per predicted
  class instead of by one global threshold only.
- Per-class pseudo-label statistics and train metrics are written during
  training.

The current implementation validates labeled class counts before using balanced
sampling, weighted CE, or minority-biased pseudo-labels. If any labeled class is
missing, training stops with a clear error instead of silently clamping the
count.

## Main Files

- `dual/train.py`: main dual training entry point.
- `dual/run_dual.ps1`: single CIFAR-10-LT dual run.
- `dual/run_dual_mode_compare.ps1`: runs the three sampler / weighted-CE ablation
  modes under the same setting.
- `dual/models/`, `dual/dataset/`, `dual/utils/`: model, dataset, and utility code for
  the dual experiment folder.

## Single Training Run

From the `dual/` folder:

```powershell
cd dual
.\run_dual.ps1
```

Equivalent direct command:

```powershell
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --imb-ratio 100 --dual-train-mode dual_sampler_weighted_ce --seed 5 --out results\dual_cifar10_imb100
```

`run_dual.ps1` uses the default `train.py` dual mode, which is currently
`dual_sampler_weighted_ce`.

## Sampler / Weighted-CE Ablation

Use `--dual-train-mode` to isolate where the minor model's minority emphasis
comes from:

- `dual`: no minor balanced sampler and no minor weighted CE.
- `dual_sampler`: minor balanced sampler only.
- `dual_sampler_weighted_ce`: minor balanced sampler plus minor weighted CE.

Run all three modes:

```powershell
.\run_dual_mode_compare.ps1
```

Default comparison settings:

- dataset: CIFAR-10
- labeled samples: 4000
- imbalance ratio: 100
- architecture: WideResNet
- epochs: 200
- eval step: 1024
- seed: 5
- output root: `results\dual_mode_compare`

You can change them with PowerShell parameters:

```powershell
.\run_dual_mode_compare.ps1 -Seeds 1,2,3 -Epochs 200 -BatchSize 64 -ResultRoot results\dual_mode_compare
```

## Important Arguments

- `--dual-train-mode`: chooses the ablation mode.
- `--major-top-ratio`: base top-k ratio for class-wise pseudo-label selection.
- `--classwise-quota`: class-wise quota strategy.
- `--minority-threshold-gamma`: increases pseudo-label quota for minority
  classes.
- `--min-threshold`: confidence floor used after class-wise top-k selection.
- `--minority-bias-strength`: strength of the minor model's prior correction.
- `--minority-supervised-gamma`: class-weight exponent for minor supervised CE.
- `--max-minority-weight`: maximum class weight for minor supervised CE.
- `--minority-bias-warmup`: warmup steps for minority bias and weighted CE.
- `--pseudo-warmup`: warmup steps for cross pseudo-supervision.
- `--unsup-warmup`: warmup steps for unsupervised FixMatch loss.
- `--ensemble-minor-weight`: minor-model weight used in evaluation ensemble.

## Outputs

Each run writes checkpoints, logs, TensorBoard events, and CSV metrics under
`--out`.

Useful files include:

- `checkpoint.pth.tar`: latest checkpoint.
- `model_best.pth.tar`: best checkpoint by validation accuracy.
- `log.txt`: training log.
- `train_metrics.csv`: epoch-level loss, pseudo-label, memory, and timing
  metrics.
- pseudo-label group scalar logs for the major and minor models.

For mode comparison, each mode is saved separately:

```text
results\dual_mode_compare\seed_5\01_dual
results\dual_mode_compare\seed_5\02_dual_sampler
results\dual_mode_compare\seed_5\03_dual_sampler_weighted_ce
```

## Notes

- Use `--expand-labels` with the provided CIFAR-10-LT setting unless you are
  intentionally testing another labeled split.
- The code reconciles expanded labeled dataset counts with the original split
  counts before computing sampler weights or class weights.
- If `--dual-train-mode dual` is selected, the minor model uses the normal
  labeled loader and unweighted supervised CE, while the rest of the dual
  pseudo-labeling logic remains active.
