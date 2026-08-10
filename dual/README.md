# Major-Minor Dual FixMatch

This folder contains the cleaned dual version only. Extra experiment runners and analysis-only code were removed.

Kept components:
- Major-Minor dual models
- Balanced labeled sampling for the minor model
- Class-weighted supervised loss for the minor model
- Minority-biased pseudo-label probabilities for the minor model
- Class-wise pseudo-label selection
- Cross pseudo-supervision between major and minor models

Run:

```powershell
cd C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual
.\run_dual.ps1
```

Equivalent direct command:

```powershell
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --imb-ratio 100 --seed 5 --out results\dual_cifar10_imb100
```

Main tunable arguments are `--major-top-ratio`, `--classwise-quota`, `--minority-threshold-gamma`, `--min-threshold`, `--minority-bias-strength`, `--minority-supervised-gamma`, `--max-minority-weight`, `--minority-bias-warmup`, `--pseudo-warmup`, `--unsup-warmup`, and `--ensemble-minor-weight`.

