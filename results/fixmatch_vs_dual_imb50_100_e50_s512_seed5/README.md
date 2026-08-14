# FixMatch vs Dual Results

CIFAR-10-LT comparison results for baseline FixMatch and the dual sampler + weighted CE method.

Settings:

- Dataset: CIFAR-10
- Labeled samples: 4000
- Imbalance ratios: 50, 100
- Seed: 5
- Training length: 50 epochs, 512 eval steps per epoch, 25,600 total steps per run
- Batch size: 32
- Mu: 7

## Overall Performance

| Imbalance ratio | Method | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 50 | baseline FixMatch | 81.56 | 81.70 |
| 50 | dual sampler + weighted CE | 88.55 | 88.61 |
| 100 | baseline FixMatch | 76.69 | 76.57 |
| 100 | dual sampler + weighted CE | 87.95 | 88.05 |

## Minority-Class Performance

Minority means classes 5-9. Tail means classes 7-9.

| Imbalance ratio | Method | Group | Recall | Precision | F1 |
|---:|---|---|---:|---:|---:|
| 50 | baseline FixMatch | minority 5-9 | 72.36 | 94.49 | 81.80 |
| 50 | dual sampler + weighted CE | minority 5-9 | 85.70 | 94.60 | 89.86 |
| 50 | baseline FixMatch | tail 7-9 | 70.80 | 97.57 | 82.01 |
| 50 | dual sampler + weighted CE | tail 7-9 | 87.93 | 97.49 | 92.46 |
| 100 | baseline FixMatch | minority 5-9 | 63.22 | 94.98 | 75.13 |
| 100 | dual sampler + weighted CE | minority 5-9 | 83.66 | 95.36 | 89.01 |
| 100 | baseline FixMatch | tail 7-9 | 58.53 | 97.88 | 72.60 |
| 100 | dual sampler + weighted CE | tail 7-9 | 86.33 | 97.70 | 91.66 |

The dual method mainly improves minority recall while keeping precision nearly unchanged.

## Files

- `summary.csv`: run-level accuracy and macro-F1 summary.
- `minority_metrics_summary.csv`: grouped minority/tail recall, precision, and F1.
- `minority_metrics_by_class.csv`: class-level recall, precision, and F1.
- `imb*/**/train_metrics.csv`: epoch-level metrics.
- `imb*/**/per_class_eval.csv`: per-class evaluation by epoch.
- `imb*/**/pseudo_stats.csv`: pseudo-label statistics.
- `imb*/dual_sampler_weighted_ce/pseudo_diversity.csv`: major/minor pseudo-label diversity statistics.
