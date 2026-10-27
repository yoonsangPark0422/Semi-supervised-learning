# Weak-augmentation held-out validation diagnosis

Completed 2026-10-26. Uses previous validation-trained real last-2 epoch50
checkpoints, seeds1/2. NOT the recent full-train checkpoints. Source IDs and
checkpoint SHA256 are in summary.json. No training or weight changes occurred.
Clean EMA predictions exactly reproduced the saved diagnostic baseline.

Each of 5000 held-out images received five weak crops/flips per seed. Masks use
the original classwise_topk_mask function and actual final cross/gate rules,
in shuffled batches of224 (last shorter batch retained). Both online and EMA
weights were examined. Ground truth is used only after target/mask selection.
This is eval-mode frozen-BN inference, not actual train-mode BN, strong-view loss
or a historical training trace. No test images were used. Seed splits overlap;
pooled counts are seed-image-augmentation observations, not independent images.

## Online weights: true Tail subset, 15000 observations
- Minor correct:13520 (90.13%).
- Self selected:14227 (94.85%); rejected below .5:321; quota:422; gate:30.
- Incorrect self targets:843 (5.93% of selected);619 are true horses.
- 205 incorrect self targets have Minor confidence>=.95.
- Major predicts the same wrong class in473/843 cases; both confidences>=.95
  and same wrong class in101 cases. Major correct in118/843.
- Only1/843 is raw-correct -> adjusted-wrong; most errors precede correction.
- Minor->Major sent:11796; predicted non-Tail:1335; below .95:1863; conflict:6.
- Incorrect sent targets among this TRUE-TAIL subset:24. This does not measure
  global exchange precision, because non-Tail images incorrectly sent are excluded.
- Minor self/cross target conflicts:44; Major conflicts:0 on true Tail subset.
- 286/3000 seed-image pairs change prediction across views;98/3000 are wrong
  and self-selected on all five views.
- Recall by true class:horse83.62%, ship92.70%, truck94.08%.

## EMA corroboration
- Self selected:14256/15000 (95.04%); wrong self746, including619 horses.
- Only1 wrong self is created by correction; Major agrees with wrong target454
  times and is correct121 times. Both confidence>=.95 and same wrong class114.
- Minor self/cross conflicts46; Major conflicts0 on true Tail subset.

## Interpretation
Widespread Tail self rejection and target conflict are not the dominant observed
events in this snapshot. High-confidence shared errors exist, especially horses
misclassified as cats/dogs/deer. Threshold lowering alone cannot fix those errors.
The prevalence does not establish the cause of DeCon's test advantage or prove
that these wrong targets historically reinforced errors. Do not retune on test.

Per-image CSVs identify repeat stability and original train-source index.
Per-observation CSVs contain all three predictions, confidence, mask decisions,
rejection reasons, and target conflicts. Repeat lists follow the recorded repeats.
