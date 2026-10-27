# 10% data comparison: DeCon vs real last dual-2

Both methods train fresh with seed1 on the same 5000 balanced unlabeled CIFAR-10
train images (500 per class), with400 labeled images included in that pool.
Unique labeled counts:162,97,58,35,21,13,7,4,2,1. Original ratio100 count-generation
rule is retained; integer allocation makes realized max/min162, not exactly100.
This rounding and altered few-shot class support limit causal attribution to
dataset size alone when comparing with the full-data experiment.

Canonical shared_indices.npz is supplied to both original loaders. Actual data
sizes and class counts are asserted for both during preflight and training.
No other CIFAR-10 train images enter the dataloaders. The underlying torchvision
loader reads the complete archive before selecting the audited subset.

Common:WRN28-2,SGD lr.03,wd.0005,EMA.999,batch32,mu7,50x512 steps,workers2.
This holds optimizer updates constant, not exposure per unique image; each image
is reused more often than in full-data training. Runtime is not 10% of baseline.
No validation holdout; official test10000 evaluated only at the final epoch.
No resume/init or selecting the best test epoch. Save final checkpoints.

Real last-2:cross ON, original Minor self floor.5, confidence weighting OFF,
hard self targets, LLA unchanged, balanced sampler alpha1.2, ensemble Minor.8.
DeCon:original method-specific losses; --match-dual-data forces canonical IDs.
DeCon's legacy num-max arguments do not control the canonical split.
Seed1 only is a preliminary comparison, especially with one labeled truck.
