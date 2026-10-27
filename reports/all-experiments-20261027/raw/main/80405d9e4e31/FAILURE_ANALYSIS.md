# Failure evidence analysis

Scope: prior held-out-validation real last-2 checkpoints, seeds1/2, online
weights, five weak views; 15000 true-Tail observations. These are dependent
repeated observations, not unique images or actual historical training losses.
Recent full-train test results establish the performance gap; this older model
diagnostic supplies hypotheses, not a proven causal explanation of that gap.

## 1. Horse errors dominate wrong self targets
Horse:4750 selected/5000 views,619 incorrect (13.03% of selected).
Ship:4696 selected,105 incorrect (2.24%).
Truck:4781 selected,119 incorrect (2.49%).
Horse wrong predictions (all819 views):cat360, deer145, dog131.
Stable wrong-and-self-selected across all5 views:98 seed-image pairs,
79 horses,8 ships,11 trucks;97 unique original train-image IDs.

## 2. Most wrong self targets are lower-confidence, but high-confidence errors remain
Selected confidence bins:
- [.5,.7):599 selected,247 wrong (41.24%).
- [.7,.9):1020 selected,279 wrong (27.35%).
- [.9,.95):626 selected,112 wrong (17.89%).
- [.95,1]:11982 selected,205 wrong (1.71%).
526/843 incorrect self targets (62.40%) lie below .9. Removing that bin also
removes1093 correct targets; this does not justify changing a threshold without
further validation. Non-Tail subset was not included in these bin statistics.

## 3. Cross policy cannot use every available correct teacher
Of843 wrong self targets:Major agrees with the wrong target473 times, differs
but is also wrong252 times, and is correct118 times. In all118 correct-Major
cases, Major->Minor is OFF. Because these are true Tail labels, a correct Major
prediction is outside the policy's Head-only sending rule. This shows a policy
coverage limitation, not proof that removing the class restriction improves learning.

## 4. Correction and opposing cross targets are not predominant here
Only1/843 wrong self targets was correct before LLA correction.31 changed class,
so most wrong targets already existed before adjustment. Among843 wrong self
cases,24 have conflicting incoming Major targets and17 receive the same wrong
target from Major. All30 true-Tail gate suppressions in this audit suppress a
correct Minor target. The latter is small but warrants a separate gate ablation.

## Threshold counterfactual, true-Tail subset only
Minor->Major threshold .95:11796 sent,24 wrong.
.90:12319 sent,41 wrong. This adds506 correct and17 wrong observations.
Cannot estimate GLOBAL exchange precision from this subset: non-Tail images
incorrectly predicted Tail must be included before deciding a threshold.

## Recommended next controlled comparison
Keep cross enabled. First examine weighting lower-confidence self targets instead
of lowering the self floor, without changing cross/gate simultaneously. Separately
evaluate whether verified reliable Major Tail predictions can correct Minor.
Neither change is implemented or claimed to improve performance by this report.
Raw aggregate data:failure_evidence.json. Per-image and per-view CSVs remain in
this directory, with validation and original train-source indices.
