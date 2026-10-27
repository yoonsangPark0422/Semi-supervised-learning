# Dual real last-2: official CIFAR-10 test evaluation

Evaluated 2026-10-25. Cross ON, hard Minor self targets, learnable LLA.
Frozen final epoch-50 checkpoint.pth.tar from each completed seed (1, 2).
EMA Major/Minor weights and saved online LLA parameters; no test-time updates.
Training used the original train split with 5,000 held-out validation images;
this evaluation does not retrain on the full train split. Official test has
10,000 images (1,000 per class), distinct from that validation split.
Command: existing trainer with --eval-only --resume <checkpoint>
--val-samples-per-class 0. Full arguments and logs are in each evaluation folder.
The loader builds unused training datasets in eval-only mode; no training is run.

| Metric | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|
| Major accuracy | 86.08 | 86.14 | 86.11 |
| Minor raw accuracy | 88.88 | 89.11 | 89.00 |
| Minor LLA accuracy | 89.40 | 89.27 | 89.34 |
| Raw ensemble accuracy | 88.66 | 88.58 | 88.62 |
| LLA ensemble accuracy | 89.02 | 88.93 | 88.98 |
| Minor LLA macro-F1 | 89.4547 | 89.3470 | 89.4008 |

All values are percentages. Means average the two seed scores on the same test
set; this is not an ensemble across seeds or 20,000 independent test images.
Per-class precision, recall, F1, and support are in per_class_eval.csv in:
- dual_real_last_2_lla_hard_official_test_e50_seed1
- dual_real_last_2_lla_hard_official_test_e50_seed2

The validation-selected Minor LLA output averages 89.35% on validation and
89.335% on official test. Cross-OFF was not evaluated on test here; its prior
comparison remains a validation comparison. No hyperparameters were changed
based on these test results.
