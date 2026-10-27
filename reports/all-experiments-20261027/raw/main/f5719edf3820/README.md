# Held-out validation Tail selection audit

Uses the prior validation-trained real_last_dual_2 seed1/2 final epoch50 EMA
predictions. Does NOT use the latest full-train model (validation contamination).
No training or parameter updates. Each seed has 1500 true Tail images.
Totals below pool seed-image evaluations, not 3000 distinct images.

This is a frozen clean-image diagnostic: apply final cross rules, and reconstruct
the actual classwise_topk_mask function over five deterministic shuffled batches
of224 across all5000 validation images. It is NOT a record of actual weak-aug
training selections, and may differ for online weights or other batch compositions.

## Cross selection: 3000 Tail evaluations
- Sent Minor->Major:2499 (83.30%). Four of these targets incorrect.
- Predicted non-Tail:254 (8.47%).
- Predicted Tail but confidence<.95:244 (8.13%).
- Simultaneous opposite-direction candidate:3 (0.10%).
- Correct prediction but not sent:237.

## Minor self selection
- Selected in all5 batch arrangements:2859 (95.30%).
- Below .5 confidence:24.
- Excluded by quota in all5 arrangements:88.
- Batch-dependent quota result:22.
- Correction gate suppressed:7 (classification priority after confidence).
- Incorrect target selected in all5 arrangements:142 (seed1:73, seed2:69).
- Of those142,118 are true horses. Major is correct for only20 of142.
- None of142 was a raw-correct -> adjusted-wrong correction flip.

## Interpretation
The earlier suspicion of widespread Tail self rejection is not supported by
this clean-validation snapshot. Both lack of cross coverage and confident
wrong self targets exist. Confident model errors, especially horse->cat/dog/deer,
need attention; confidence-only filtering cannot remove every such error.
No causal claim about historical training failure follows from this audit.
CSV files identify each sample by seed and validation_index matching the saved
epoch050 samples.csv; summary.json provides class-specific counts.
