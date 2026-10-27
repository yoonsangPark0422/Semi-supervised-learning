# Cross exchange ablation

Compare existing cross-ON runs against `--disable-cross` with seed 1 only,
CIFAR-10, 50 epochs (25,600 steps), and 500 validation images per class.
`run_no_cross_e50.ps1` uses the same arguments as the completed hard runs,
with only `--disable-cross` added and separate output paths.

The flag sets lambda_cross to zero and zeroes both exchange masks before
the correction gate. Minor's baseline self selection is therefore restored;
its hard targets, denominator, and self confidence selection stay unchanged.
Major self loss and both supervised losses are unchanged. LLA still learns
from labeled targets and all unlabeled soft targets, without a confidence filter.
Default cross-ON behavior is unchanged. Checkpoint variants distinguish ON/OFF.
Diagnostics mark actual exchanges and self suppression as disabled.

Evaluate each model separately (raw and adjusted Minor), macro-F1, and
Head/Middle/Tail recall using the same epoch and paired seed. Validation is
for comparison; it is not the final test set. This measures the total effect
of the exchange policy including its self-suppression gate, not the isolated
effect of either exchange direction. Directional ablations are not scheduled.
