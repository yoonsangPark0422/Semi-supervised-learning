# Major-Minor Dual FixMatch

## File

- `dual/train.py`

## Purpose

This is the main proposed algorithm. It trains two FixMatch-style models at the same time: a `major` model and a `minor` model.

## Model Structure

- `major` model: standard FixMatch-style branch.
- `minor` model: minority-aware branch.
- Cross pseudo-supervision: each branch can train on pseudo-labels produced by the other branch.
- Evaluation writes metrics for:
  - `major`
  - `minor`
  - `major+minor` ensemble

## Key Mechanisms

- Minority-biased pseudo-label probabilities for the minor branch.
- Class-wise pseudo-label top-k selection.
- Optional balanced sampler for the minor labeled batches.
- Optional class-weighted supervised CE for the minor branch.
- Warmups for pseudo-label use, unsupervised loss, and minority weighting.

## Important Mode

The full proposed method is:

```text
dual_sampler_weighted_ce
```

This means:

- major branch: normal sampler and normal CE.
- minor branch: balanced sampler and weighted CE.
- cross pseudo-supervision is used.

## Why It Exists

The algorithm is designed to reduce majority-class bias in long-tailed semi-supervised learning, especially improving minority and tail-class F1.