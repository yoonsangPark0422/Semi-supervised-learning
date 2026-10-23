# last-2 dual

CIFAR-10 seed 1, 50 epochs x 512 steps. Separate experimental variant of last dual.
This is a narrow ablation, NOT the full target-selection proposal.

## Only training change

On a sample already accepted for Major-to-Minor exchange under the original
confidence, head/tail, warmup, conflict-exclusion and optional skip rules:
if Major and raw Minor agree on a Head label but corrected Minor changes to Tail,
disable the Minor self pseudo-label loss for that sample.
The original Major-to-Minor cross loss supplies the common Head target.
Major self loss, all cross masks/losses, other samples and logit correction remain unchanged.
Do not resolve previously excluded two-way conflicts or introduce general abstention.
No ground truth is used in this gate.

The Minor self loss uses the original selected-mask count BEFORE suppression
as its denominator. This removes conflicting contributions without increasing
the per-sample weights of remaining examples. Its total magnitude can decrease;
this is intentional and must be considered when interpreting the ablation.

## Diagnostics and limits

correction_gate_stats.csv records sampled occurrences per epoch (not unique images):
samples_seen, gate_selected, conflicting_self_suppressed,
baseline_minor_self_selected.
Existing three-prediction validation reports are preserved; samples.csv additionally
records gate eligibility and the selected Minor target.
Validation eligibility is NOT proof that the batch-dependent self-loss mask was active.
Original corrected predictions remain in the reports for honest pre/post comparison.
Use held-out validation only; test is not used for rule selection.
EMA clean-image validation differs from online augmented training.

## Run

run_seed1_e50.ps1 uses the existing local CUDA environment and data cache.
It refuses an existing result directory. No checkpoint resume: train from scratch
with the 50-epoch schedule, same seed, split and hyperparameters as last dual.
Outputs include best/final checkpoints, validation metrics and disagreement records.
Hardware/path configuration is local; adjust paths when running on another machine.
Synthetic policy and loss tests: python -B test_correction_gate.py.
