# External Baseline Comparison Helper

## File

- `scripts/compare_external_baselines.py`

## Purpose

This helper evaluates external baselines with the same metric format used for the dual experiments.

## Metrics

It is intended to produce or align metrics such as:

- Accuracy
- Macro F1
- Minority-class recall / precision / F1
- Tail-class recall / precision / F1
- Per-class evaluation CSV files

## Use Case

Use this for comparing methods such as DeCon against the proposed dual method under matched dataset, imbalance ratio, seed, training-step, and evaluation settings.