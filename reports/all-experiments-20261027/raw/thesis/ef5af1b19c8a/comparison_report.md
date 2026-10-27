# CIFAR-10 Baseline Comparison

| Method | Epochs | Latest Acc | Latest Macro F1 | Best Epoch | Best Acc | Best Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| crest | 100 | 91.19 | 91.19619131088257 | 99 | 91.19 | 91.19033813476562 |
| daso | 100 | 91.57 | 91.57692790031433 | 99 | 91.6 | 91.61182641983032 |
| abc | 100 | 82.03 | 82.1807861328125 | 98 | 82.43 | 82.62916803359985 |
| proposed | 100 | 93.47 | 93.48114132881165 | 100 | 93.47 | 93.48114132881165 |

Generated from each method's train_metrics.csv under result/seed_5.
Per-class accuracy/F1 and pseudo-label statistics are saved in each method folder.
