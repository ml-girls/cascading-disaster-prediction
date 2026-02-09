# Data Preparation Pipeline (`prepare_data.py`)

This script handles the transformation of raw labeled data into feature matrices suitable for machine learning models.

## Usage

```bash
python prepare_data.py --split_type [random|chronological] --filter_cascades [True|False]
```

### Arguments

- `--split_type`: 
    - `chronological`: Splits data based on time (train on past, test on future). Best for realistic evaluation.
    - `random`: Randomized split with stratification based on the presence of any cascade.
- `--filter_cascades`:
    - `True`: Filters the dataset to ONLY include events that triggered at least one cascade. (Useful for multiclass or conditional models).
    - `False`: Keeps all events. (Required for binary "is_cascade" detection).

## Output Structure

Depending on the flags, data is saved to specific subdirectories:

- `random_data/` or `chronological_data/` (if `filter_cascades=False`)
- `random_filtered_data/` or `chronological_filtered_data/` (if `filter_cascades=True`)

Each directory contains:
- `X_train.npy`, `X_test.npy`: Feature matrices.
- `y_train.npy`, `y_test.npy`: Target matrices (multilabel).
- `metadata.pkl`: Stores `feature_names`, `target_names`, and the `split_type`.

## Feature Engineering Components

- `features.py`: Contains base feature engineering (Temporal, Impact, Spatial, etc.).
- `aggregate_features.py`: Handles fitting and transforming aggregate statistics (Location/State cascade rates and transition probabilities).
