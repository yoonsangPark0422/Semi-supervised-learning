# Fixed Minor-Balanced-Weighted Dual

## File

- `dual/train_minor_weighted_balanced.py`
- `dual/run_minor_weighted_balanced.ps1`

## Purpose

This file fixes the full minor-branch setting so it always uses balanced sampling and weighted CE.

## Important Clarification

Despite the file name, this is not a true single-model minor-only method.

It still uses:

- a major model,
- a minor model,
- cross pseudo-supervision,
- major/minor/ensemble evaluation.

The difference from `dual/train.py` is that the minor branch options are fixed on instead of being selected through `--dual-train-mode`.

## Relationship To The Main Method

This is essentially a fixed version of:

```text
dual_sampler_weighted_ce
```

Use it as a convenience or reproduction script, not as a separate algorithm claim.