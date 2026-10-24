# dual real last: Minor LLA + soft pseudo-label self loss

Minor에만 LLA 보정 학습과 soft 자기 의사 라벨 loss를 적용한 Dual 변형입니다.
Major, balanced sampler, weighted CE, 교환 기준과 last-2 충돌 처리는 유지합니다.
논문 전체 재현이 아니며, 아래 차이점과 초기화/최적화 설정을 명시했습니다.
정식 50 epoch 학습은 아직 실행하지 않았습니다.

Separate variant of last-2 dual. Existing folders/results are not modified.
This is an independently implemented **Dual integration of the LLA core**,
NOT the official authors' code or a full reproduction of ICCV 2025 results.

## Changes and retained components

- Major training unchanged.
- Minor fixed labeled-prior logit correction is REPLACED by
  z_adjusted = z - log_softmax(lambda), with K learned parameters.
- Minor self loss uses detached soft pseudo-label probabilities, not argmax CE.
- Minor balanced sampler, weighted supervised CE, original self-selection mask,
  warmup and original hard cross-exchange rules are retained.
- The last-2 gate still suppresses the Minor self loss on accepted Major-to-Minor
  samples where the original common Head prediction changes to Tail after correction.
  Its denominator is the pre-suppression selected-self count.
- No EFCC. No new general abstention or confidence-ratio weighting.
- The old --minority-bias-strength option is removed. --minority-bias-warmup still
  controls inherited supervised-weight/threshold schedules, NOT the learned bias.
- The LLA module requires T=1; its correction is active from the first step.

## LLA learning and gradients

LLA sees the same Minor labeled batch (including its existing sampling scheme)
and Minor unlabeled weak/strong logits. No extra validation labels are used.
For each class compute the weighted sum of negative log probabilities divided by
that class's target mass plus epsilon; then average over ALL K classes:
one-hot ground truth on labeled data (Eqs. 11-12), soft target mass on unlabeled
data (Eqs. 14-15). Missing labeled classes contribute zero.
Do NOT multiply this auxiliary loss by the Minor weighted-CE weights.
Use every unlabeled sample for the LLA class loss, independent of base loss masks.

LLA loss = labeled class-average + unlabeled class-average.
CNN logits are detached for this loss; only lambda receives its gradients.
The weak-view adjusted soft targets are detached for both the auxiliary and
base losses. Base Dual loss updates the CNNs only, never lambda.
Both objectives use the same pre-update predictions at a step.
There is no unlabeled ground-truth input to LLA or target selection.

## Explicit integration choices / limitations

The public paper and supplement were checked; an official code repository was
not located. Do not present implementation choices as verified author-code defaults:
- lambda initializes to zeros (uniform softmax, no relative correction).
- Lambda optimizer: Adam, learning rate 0.002 (--lla-lr), weight decay 0.
  The supplement reports Adam at 0.002 but does not separately specify every
  lambda optimizer detail; these are explicit integration settings.
- epsilon 1e-8 (--lla-eps).
- stop-gradient for pseudo-label targets.
- no EMA for lambda; validation uses current lambda with EMA CNNs when enabled.
- Original Dual CNN optimizers, sampling, masks and 50-epoch schedule remain.
  The paper's FixMatch experiments use soft targets without confidence filtering,
  EFCC, different batch sizes/optimizer/training duration. These are NOT reproduced.
- Supported datasets: CIFAR-10 and SVHN. CIFAR-100 is rejected because the inherited
  loader does not implement the required held-out validation correctly.
- Single-process training only; distributed mode is explicitly rejected.

## Evaluation and checkpoints

Both unadjusted and LLA-adjusted metrics are reported:
- major, minor, ensemble: existing RAW baseline-comparable evaluation.
- minor_lla, ensemble_lla: apply learned Minor correction at evaluation (Eqs. 20-21).
per_class_eval.csv and TensorBoard distinguish all five names.
evaluation_metrics.csv records loss, accuracy and macro F1 for each named variant.
train_metrics.csv and best checkpoint selection keep RAW ensemble accuracy,
so evaluation changes are not silently mistaken for training improvements.
The adjusted ensemble combines Major raw logits and Minor adjusted logits with
the unchanged ensemble weight; it is a Dual adaptation, not a paper ensemble.
Diagnostics preserve Major / raw Minor / adjusted Minor predictions.
minor_selected_target is only a descriptive argmax, NOT the soft training target.

Checkpoints save lambda, its optimizer and settings alongside both CNNs/EMA states.
--resume rejects old checkpoints without LLA state. --init-checkpoint is a
weights-only warm start with new lambda/optimizers, NOT exact continuation.
--eval-only requires --resume and also supports validation diagnostics.
Checkpoint resumption does not promise bitwise replay of dataset sampler RNG.
Never use a full-train checkpoint to fit rules on a subset it has already seen.

## Files and execution

- lla.py: paper-core equations and isolated gradient paths.
- test_lla.py: formula, missing-class, gradient, mask, diagnostics, checkpoint tests.
- train_head_tail_exchange_sampler_alpha_val_split.py: integrated Dual trainer.
- run_seed1_e50.ps1: local CIFAR-10 seed 1 / 50 x 512 / val 500 per class launcher.
  Refuses existing outputs. Paths are local; adjust for other machines.
- lla_metrics.csv: per-epoch auxiliary losses, lambda, softmax(lambda), relative biases.
- correction_gate_stats.csv and validation_disagreement/: inherited/extended reports.

Run tests: python -B test_lla.py
Use --help for training options. The launcher is supplied but is NOT automatically run.
Validation is disjoint from the train pool; final test is not used to fit LLA.

## Verification (2026-10-24)

- Eight unit tests passed, including isolated lambda/CNN gradients and soft targets.
- CUDA one-step training on cached CIFAR-10 passed with a tiny held-out split.
- A second smoke step resumed CNN/LLA optimizer states; lambda optimizer step 2
  and nonzero learned parameter changes were verified.
- The resumed smoke test deliberately lowered self thresholds to exercise a
  nonempty soft-self mask. These are test-only settings, NOT launcher defaults.
- All five evaluation variants and validation diagnostics were generated.
- These smoke runs are software checks, NOT meaningful performance results.

## Sources

Lee, Park, Kim. Learnable Logit Adjustment for Imbalanced Semi-Supervised Learning
under Class Distribution Mismatch. ICCV 2025.
- Main paper, Sec. 4.2 / Eqs. 9-15; Sec. 4.4 / Eqs. 20-21:
  https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Learnable_Logit_Adjustment_for_Imbalanced_Semi-Supervised_Learning_under_Class_Distribution_ICCV_2025_paper.pdf
- Supplement, Algorithm 1 and Sec. F:
  https://openaccess.thecvf.com/content/ICCV2025/supplemental/Lee_Learnable_Logit_Adjustment_ICCV_2025_supplemental.pdf
