# Algorithm Guide

This document indexes the algorithm and experiment files currently included in this repository.

## Algorithms

| File | Purpose |
|---|---|
| [`../train.py`](../train.py) | Original FixMatch baseline. |
| [`../dual/train.py`](../dual/train.py) | Main Major-Minor Dual FixMatch implementation and sampler / weighted-CE ablations. |
| [`../dual/train_minor_weighted_balanced.py`](../dual/train_minor_weighted_balanced.py) | Fixed full-dual variant where the minor branch always uses balanced sampling and weighted CE. |
| [`../dual/train_exchange_direction_ablation.py`](../dual/train_exchange_direction_ablation.py) | Cross pseudo-supervision direction ablation. |
| [`../scripts/compare_external_baselines.py`](../scripts/compare_external_baselines.py) | External baseline evaluation helper. |

## Detailed Notes

- [FixMatch baseline](algorithms/fixmatch_baseline.md)
- [Major-Minor Dual FixMatch](algorithms/major_minor_dual.md)
- [Sampler / Weighted-CE ablation](algorithms/sampler_weighted_ce_ablation.md)
- [Fixed minor-balanced-weighted dual](algorithms/fixed_minor_weighted_balanced_dual.md)
- [Exchange-direction ablation](algorithms/exchange_direction_ablation.md)
- [External baseline comparison helper](algorithms/external_baseline_comparison.md)

## Not Included Yet

The true single-model minor-only experiment, where the major model is removed entirely, is not included in the current GitHub version yet. The current `train_minor_weighted_balanced.py` file is still a dual-model training script.