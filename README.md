# DmpR-Po reproducibility package

This package performs inference for the final full-sequence MLP and Residual CNN ensemble.

## Locked dataset and models

- Batch 1: 11,940 non-WT sequences of length 156 bp.
- Frozen split: Train 8,358; development 1,791; held-out Test 1,791.
- Each architecture is the mean prediction of three fixed seed checkpoints.
- Final ensemble: 0.5 x MLP plus 0.5 x Residual CNN.
- MLP input: 624 position-specific one-hot features plus 256 normalized 4-mer frequencies, transformed with the supplied frozen StandardScaler.
- Residual CNN input: raw 4 x 156 one-hot tensor.

## Reproduce held-out Test metrics

Install dependencies with pip install -r requirements.txt, then run:

    python scripts/reproduce_test_metrics.py

The command performs inference only. It does not train, tune, split, or select any model. It writes reproduced_test_predictions.csv and compares the ensemble with the archived locked Test predictions.

## Batch 2

The included Batch 2 cross-batch evaluation dataset contains non-WT sequences derived from a separately transformed and processed Batch 2 using the same preconstructed plasmid-library pool. Run:

    python scripts/reproduce_batch2_metrics.py

## Optional analyses

Interpretability code is deliberately excluded from this minimal inference package and can be released separately with the required input manifests.
