# Major-Minor Dual FixMatch

This folder contains the cleaned dual version only. Extra experiment runners and analysis-only code were removed.

Kept components:
- Major-Minor dual models
- Balanced labeled sampling for the minor model
- Class-weighted supervised loss for the minor model
- Minority-biased pseudo-label probabilities for the minor model
- Class-wise pseudo-label selection
- Cross pseudo-supervision between major and minor models

Run one training:

```powershell
cd C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual
.\run_dual.ps1
```

Equivalent direct command:

```powershell
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --imb-ratio 100 --dual-train-mode dual_sampler_weighted_ce --seed 5 --out results\dual_cifar10_imb100
```

Main tunable arguments are `--major-top-ratio`, `--classwise-quota`, `--minority-threshold-gamma`, `--min-threshold`, `--minority-bias-strength`, `--minority-supervised-gamma`, `--max-minority-weight`, `--minority-bias-warmup`, `--pseudo-warmup`, `--unsup-warmup`, and `--ensemble-minor-weight`.
## Sampling and Weighted CE Comparison

Use `--dual-train-mode` to isolate whether minority emphasis comes from the sampler, the supervised loss, or both:

- `dual`: no minor balanced sampler, no minor weighted CE
- `dual_sampler`: minor balanced sampler only
- `dual_sampler_weighted_ce`: minor balanced sampler plus minor weighted CE

Run all three modes:

```powershell
.\run_dual_mode_compare.ps1
```


