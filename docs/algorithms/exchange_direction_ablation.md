# Exchange-Direction Ablation

## File

- `dual/train_exchange_direction_ablation.py`
- `dual/run_exchange_direction_ablation.ps1`

## Purpose

This ablation tests whether cross pseudo-supervision direction matters.

## Modes

| Mode | Meaning |
|---|---|
| `none` | No cross exchange. Each branch uses its own pseudo-labels. |
| `major_to_minor` | Major pseudo-labels supervise the minor branch. |
| `minor_to_major` | Minor pseudo-labels supervise the major branch. |
| `bidirectional` | Both branches exchange pseudo-labels. This is the original full-dual direction. |

## What It Tests

This tells whether the improvement comes from:

- simply having two branches,
- the major branch helping the minor branch,
- the minor branch helping the major branch,
- or bidirectional pseudo-label exchange.