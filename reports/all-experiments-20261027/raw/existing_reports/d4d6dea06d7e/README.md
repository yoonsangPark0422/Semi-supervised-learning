# Held-out validation results (alpha=1.2)

These metrics come from the final epoch on a held-out validation split created
from each dataset training split. They are not official-test results.

## Protocol

- datasets: CIFAR-10 and SVHN
- labeled samples: 4,000
- imbalance ratio: 100
- validation samples: 500 per class from the training split
- training: 25,600 steps (50 x 512)
- Dual and Minor-only sampler exponent: alpha=1.2
- reported checkpoint: final epoch

## CIFAR-10-LT

| Method | Seeds | Accuracy | Macro F1 | Minority 5-9 F1 | Tail 7-9 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dual ensemble | 5 | 89.4360 +/- 0.6534 | 89.4749 +/- 0.6403 | 90.4097 +/- 0.5894 | 93.9659 +/- 0.2065 |
| Minor-only | 5 | 88.8400 +/- 0.7397 | 88.8969 +/- 0.7376 | 90.2469 +/- 0.7279 | 93.1279 +/- 0.1727 |
| DeCon | 3 | 82.2800 +/- 0.4779 | 82.2929 +/- 0.4885 | 81.1138 +/- 0.7422 | 83.4071 +/- 0.2858 |

## SVHN-LT

| Method | Seeds | Accuracy | Macro F1 | Minority 5-9 F1 | Tail 7-9 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dual ensemble | 5 | 94.8240 +/- 0.7006 | 94.8321 +/- 0.7038 | 94.6294 +/- 1.0197 | 94.6120 +/- 1.0445 |
| Minor-only | 5 | 93.8400 +/- 0.4966 | 93.8412 +/- 0.5140 | 93.8770 +/- 1.1259 | 93.6071 +/- 0.9237 |
| DeCon | 5 | 91.6920 +/- 0.2386 | 91.6743 +/- 0.2454 | 89.8640 +/- 0.2423 | 89.5653 +/- 0.3791 |

Values are mean +/- sample standard deviation. CIFAR-10 DeCon has three seeds;
all other rows have five. This difference is recorded explicitly so the table
does not imply a five-seed DeCon result that was never run.

## Included data

- `aggregate_summary.csv`: mean and sample standard deviation by dataset and method
- `run_summary.csv`: final-epoch metrics for every included seed

The CSV metrics were reconstructed from the final-epoch `per_class_eval.csv`
of each run. Every included row had ten classes with support 500 per class.
