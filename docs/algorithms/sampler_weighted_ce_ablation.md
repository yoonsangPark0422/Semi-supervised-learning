# Sampler / Weighted-CE Ablation

## File

- `dual/train.py`
- `dual/run_dual_mode_compare.ps1`

## Purpose

This ablation isolates which part of the minor branch improves minority and tail performance.

## Modes

| Mode | Minor balanced sampler | Minor weighted CE | Meaning |
|---|---:|---:|---|
| `dual` | No | No | Dual structure only. |
| `dual_sampler` | Yes | No | Tests balanced labeled exposure. |
| `dual_weighted_ce` | No | Yes | Tests class-weighted supervised loss. |
| `dual_sampler_weighted_ce` | Yes | Yes | Full proposed minor-branch training. |

## Interpretation

This ablation answers whether performance comes from:

- the dual structure itself,
- balanced sampling,
- weighted CE,
- or the combination of balanced sampling and weighted CE.

Previous completed results showed that `dual_sampler_weighted_ce` is the strongest configuration.