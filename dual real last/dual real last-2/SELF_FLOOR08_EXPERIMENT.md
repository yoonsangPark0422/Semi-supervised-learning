# Minor self threshold floor .8 ablation

Compare to dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1.
Only --min-threshold changes from .5 to .8. Confidence weighting flag is absent
(False); selected self CE weights are one. Cross and correction gate stay ON;
Major self threshold .95; both cross thresholds .95; LLA losses unchanged.

Inherited schedule: global steps0..511 have pseudo self/cross losses disabled.
At step512 the Minor confidence floor is .95. It decreases linearly over2048
steps, reaching .8 at step2560, then stays .8. Global-step units, not epochs.
Classwise top-k selection still applies in addition to this floor.

Seed1,50x512 steps, fresh weights;500 validation images/class excluded from both
training pools. Same images, batch32/mu7/lr.03/sampler1.2/ensemble.7/workers2 as
baseline. Evaluate validation each epoch; no official test evaluation. Compare
epoch50 accuracy/macro-F1/Tail and horse recall. Save full arguments and source
snapshots in separate output. No existing checkpoint is resumed or overwritten.
