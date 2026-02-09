# XGBoost Training Scripts

This directory contains scripts for training and evaluating XGBoost models using the prepared cascade datasets.

## Requirements

Ensure you have run the data preparation pipeline first:
```bash
# Example
python feature_eng/prepare_data.py --split_type random --filter_cascades False
```

## Model Types

Each script implements a different classification strategy:

### 1. Binary Detection (`xgboost_binary.py`)
- **Task**: Predicts if an event will trigger **ANY** cascade.
- **Data Requirement**: Use `random_data/` or `chronological_data/` (not filtered).
- **Run**:
  ```bash
  python xgboost_binary.py --data_dir random_data
  ```

### 2. Multiclass Selection (`xgboost_multiclass.py`)
- **Task**: Predicts the **MOST LIKELY** single secondary event triggered.
- **Data Requirement**: Use `random_filtered_data/` or `chronological_filtered_data/` (cascades only).
- **Run**:
  ```bash
  python xgboost_multiclass.py --data_dir random_filtered_data
  ```

### 3. Multilabel Prediction (`xgboost_multilabel.py`)
- **Task**: Predicts **ALL** secondary events triggered (Binary Relevance).
- **Data Requirement**: Use `random_filtered_data/` or `chronological_filtered_data/` (cascades only).
- **Run**:
  ```bash
  python xgboost_multilabel.py --data_dir random_filtered_data
  ```

### 4. Logistic Regression Baseline (`logreg_train.py`)
- **Task**: Simple linear baseline for binary cascade detection.
- **Run**:
  ```bash
  python logreg_train.py --data_dir random_data
  ```

## Outputs

Each script saves results to the `models/` directory:
- `binary_cascade_model.json` / `multiclass_model.json` / `multilabel_models/`
- `metadata.pkl`: Stores feature names and training parameters.
- `results.txt`: Comprehensive performance metrics (F1, Precision, Recall, etc.).
- `plots/`: (Optional) Visualizations like confusion matrices.
