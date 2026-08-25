# FixMatch Baseline

## File

- `train.py`

## Purpose

This is the original single-model FixMatch baseline. It trains one classifier using labeled data and unlabeled data with weak/strong augmentation consistency.

## Model Structure

- One model only.
- No major/minor branch separation.
- No cross pseudo-supervision.
- No minority-specific balanced sampler.
- No minority-specific weighted CE.

## Data Usage

- Labeled set: imbalanced labeled subset from CIFAR training data.
- Unlabeled set: CIFAR training data used as unlabeled pool.
- Evaluation: official CIFAR test split in the existing reported experiments.

## Role In Experiments

Use this as the base SSL baseline. It shows the majority-class bias problem under long-tailed labeled data.