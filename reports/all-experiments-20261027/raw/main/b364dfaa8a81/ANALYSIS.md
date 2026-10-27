# Reduced-data comparison analysis, 2026-10-27

Same400 labeled/5000 balanced unlabeled images, seed1,50x512 updates, official
test10000. Labels are included in the unlabeled pool. Counts162,97,58,35,21,13,7,4,2,1;
requested ratio100 rounds to realized162. Identical actual data audits passed.
Only one labeled truck; seed and sample-selection variability are not measured.

## Final official-test results (%)
| Output | Accuracy | Macro-F1 | Head recall | Middle recall | Tail recall |
|---|---:|---:|---:|---:|---:|
|DeCon|53.85|51.22|77.90|52.45|31.67|
|Real last-2 raw Minor|65.67|65.96|80.37|48.45|73.93|
|Real last-2 LLA Minor|67.50|67.57|77.37|52.98|77.00|
|Real last-2 LLA ensemble|67.35|67.49||||

Of1365 additional correct predictions for LLA Minor vs DeCon,1360 occur in
Tail classes (horse+318,ship+206,truck+836);Head-16,Middle+21.
Full-data accuracy:DeCon90.72 vs LLA Minor89.43. Degradation36.87 vs21.93 points.
Data size is not the only exact change:integer imbalance and selected IDs differ;
same update budget reuses each unique image far more often.

## Post-hoc frozen DeCon diagnostic (not retuning)
Reproduced official scores exactly with original batch32, EMA weights, tau_b2.
Balanced raw head52.69%, adjusted53.85%;standard head52.66%.
Final adjustment changes724 predictions:282 wrong->right,166 right->wrong,
net+116 correct (+1.16 points). Thus evaluation adjustment helps rather than
causing the overall collapse.
Raw balanced head predicts truck only9 times across all10000 test images,
with2 correct trucks. Adjusted output predicts truck173 times,56 correctly.
Among1000 actual trucks,842 are classified as automobile after adjustment.
Full-data truck recall95.7%;small-data5.6%.
Full-data qbal is near uniform (~.094-.104);small-data qbal automobile.18575,
horse.04403,ship.05346,truck.007983, despite true train-pool frequencies.10 each.
qbal is an EMA of model predictions, not a measurement of true ground-truth
distribution. Bias may both reflect and interact with classifier errors; the
checkpoint alone cannot establish causal direction or instability over time.
Raw details:decon_diagnostic.json.

## Real last-2 preserves Tail beyond final LLA correction
Raw Minor65.67%, LLA67.50% (+1.83 points). Raw truck recall86.7%, adjusted89.2%.
This advantage exists before final correction, so cannot be attributed to
inference-time LLA alone. Weighted CE, balanced sampling, exchange and LLA
training are bundled; ablations are required to identify their contributions.

## Exchange quality remains a weakness
Final-epoch training diagnostics count repeated sampling, not unique images:
Major->Minor:47082 sent,28066 correct (59.61%), vs full-data92.28%.
Minor->Major:29727 sent,25881 correct (87.06%), vs full-data99.15%.
No-confidence calibration guarantee:high-threshold exchange can remain wrong.
Without reduced-data cross-OFF control this does not prove exchange is harmful.
LLA Minor airplane precision47.74%, cat recall40.8%, dog recall40.6% remain weak.

## Claims supported and not supported
Supported:this seed's low-data setting favors real last-2, almost entirely via
Tail preservation. DeCon's learned classifiers strongly under-recognize truck,
and its predicted class distribution is correspondingly biased.
Not established:general superiority across seeds, distribution estimation as
the root cause, LLA alone as the reason, or absence/presence of overfitting.
Next most informative replication is independent seeds with both methods sharing
the same per-seed reduced split; do not tune method parameters on these test scores.
