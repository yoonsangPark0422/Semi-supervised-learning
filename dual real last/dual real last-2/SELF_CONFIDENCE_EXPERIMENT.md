# Minor self-confidence weighting ablation

Enabled only with --minor-self-confidence-weight; default baseline unchanged.
For each selected unlabeled sample, multiply Minor hard self CE by its detached
LLA-adjusted weak confidence c. Denominator stays the baseline self mask count
before gate suppression, not the sum of c. Thus .6 contributes .6 of its original
self term and .95 contributes .95. No gradient flows into confidence/LLA through
this loss. All predicted classes are eligible; no ground-truth class is used.

Major self, cross targets/masks/weights, Minor correction gate, LLA soft class
losses (all unlabeled), supervised losses, and classwise selection stay unchanged.
The change also reduces overall Minor self loss magnitude; any improvement cannot
be attributed solely to relative sample weighting without a constant-scale control.
LLA confidence is an uncalibrated score, not a guaranteed probability of correctness.

Run seed1,50x512 steps, validation500/class held out from labeled AND unlabeled,
fresh initialization. Exact baseline: dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1.
Same batch32/mu7/lr.03/sampler1.2/ensemble.7, workers2 and data seed.
Do not compare this validation result directly to recent full-train test results.
Do not evaluate official test during this ablation. Compare final epoch50 Minor
raw/LLA accuracy, macro-F1, horse/Tail recall, and weak-view selection diagnostics.
Checkpoint variant suffix prevents accidental baseline/weighted resume mixing.
