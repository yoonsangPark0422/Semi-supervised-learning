# Final official-test results (alpha=1.2)

These metrics were generated from the final epoch on the official test split
after validation-based hyperparameter selection. Values are mean +/- sample
standard deviation over seeds 1, 2, and 3.

## CIFAR-10-LT, imbalance ratio 100

| Method | Accuracy | Macro F1 | Minority 5-9 F1 | Tail 7-9 F1 |
| --- | ---: | ---: | ---: | ---: |
| Dual ensemble | 88.8100 +/- 0.2651 | 88.8652 +/- 0.2564 | 90.1291 +/- 0.3044 | 93.0051 +/- 0.4547 |
| Minor-only | 88.5133 +/- 0.4000 | 88.5683 +/- 0.3964 | 90.1894 +/- 0.5254 | 92.3381 +/- 0.5612 |
| DeCon | 83.6033 +/- 0.2136 | 83.5737 +/- 0.2187 | 83.1315 +/- 0.1772 | 85.2277 +/- 0.1278 |

## SVHN-LT, imbalance ratio 100

| Method | Accuracy | Macro F1 | Minority 5-9 F1 | Tail 7-9 F1 |
| --- | ---: | ---: | ---: | ---: |
| Dual ensemble | 95.6310 +/- 0.3111 | 95.1315 +/- 0.4092 | 94.3129 +/- 0.6609 | 93.3028 +/- 0.7638 |
| Minor-only | 94.9844 +/- 0.2806 | 94.6884 +/- 0.3158 | 94.1554 +/- 0.2504 | 92.9314 +/- 0.2602 |
| DeCon | 92.2263 +/- 0.2470 | 91.0050 +/- 0.2214 | 87.5768 +/- 0.1946 | 85.0899 +/- 0.2589 |

## Class-level official-test metrics

The original final-epoch class-level CSV files are included under
`per-class/<dataset>/<method>/seed_<n>.csv` for both datasets, all three
methods, and seeds 1, 2, and 3.

```text
per-class/
  cifar10/{dual,minor,decon}/seed_{1,2,3}.csv
  svhn/{dual,minor,decon}/seed_{1,2,3}.csv
```

The Dual files contain separate rows for the jointly trained `major` and
`minor` branches and for the reported `ensemble` output. The Minor-only and
DeCon files contain their respective single-model rows. Each row identifies
the final epoch, model, class index, class accuracy/recall, F1, and support.
Dual and Minor-only files also store precision and recall explicitly; the
original DeCon output stores accuracy, F1, and support.

## Included data

- `aggregate_summary.csv`: three-seed mean and standard deviation
- `run_summary.csv`: per-seed final metrics and original output paths
- `per-class/`: original per-class final metrics for all 18 official-test runs

The reported Dual result is the ensemble output. Within the jointly trained
Dual model, the Minor branch alone reached 88.9767% mean accuracy on CIFAR-10
and 95.6400% on SVHN; these branch values are diagnostic results, not the
pre-declared primary output.
