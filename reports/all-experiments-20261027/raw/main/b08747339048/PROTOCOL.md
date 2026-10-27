# Fresh matched three-method CIFAR-10 comparison

Seed 1 for each method; all models trained from scratch, without resume/init.
Methods: new_dual (original official-test runner's source), real_last_dual_2
(LLA, hard self, cross ON), DeCon (Dual-matched data loader).

Common: WRN-28-2, SGD lr .03, weight decay .0005, EMA .999, batch32, mu7,
50x512 updates, labeled4000 expanded to20000, ratio100, full balanced unlabeled
train pool50000 including labeled images, no validation holdout. Exact labeled
and unlabeled IDs verified by split_audit.json. Official test10000 evaluated
only after epoch50; no best-epoch selection or test-based tuning. Two Dual
ensembles use Minor weight .8. Single-seed results are preliminary.

Model-specific losses/samplers/adjustments remain part of each method; common
conditions do not mean identical model internals, compute cost, or random draw
sequence. Report Major/Minor raw/ensemble for new_dual, these plus LLA-adjusted
Minor/ensemble for real_last_dual_2, and DeCon's native output. Compare raw
ensembles as well as each method's prescribed output, without choosing an
output retrospectively based on test accuracy. Save accuracy, macro-F1 and
per-class precision/recall/F1. This experiment supersedes mixed-condition
historical comparisons; it does not reuse their result files.
