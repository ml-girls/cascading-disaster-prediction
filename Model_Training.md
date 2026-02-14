# XGBoost Training Scripts

This directory contains scripts for training and evaluating XGBoost models using the prepared cascade datasets.

## Requirements

1.  **Run the data preparation pipeline** first to generate serialized features:
    ```bash
    python feature_eng/prepare_data.py --split_type random --filter_cascades False
    ```
2.  **Input Verification**: Scripts now automatically verify the presence of `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and `metadata.pkl` before execution.

## Model types

### 1. Binary Detection (`xgboost_binary.py`)
- **Task**: Predicts if an event will trigger **ANY** cascade.
- **Optimization**: Automatically finds the **optimal decision threshold** to maximize the F1 score on the test set.
- **Run**:
  ```bash
  python xgboost_binary.py --data_dir random_data
  ```

### 2. Multiclass Selection (`xgboost_multiclass.py`)
- **Task**: Predicts the **MOST LIKELY** single secondary event triggered.
- **Metrics**: Reports **Top-3 and Top-5 accuracy** to account for complex multi-event possibilities.
- **Run**:
  ```bash
  python xgboost_multiclass.py --data_dir random_filtered_data
  ```

### 3. Multilabel Prediction (`xgboost_multilabel.py`)
- **Task**: Predicts **ALL** secondary events triggered using a Binary Relevance strategy.
- **Optimization**: Performs **per-label threshold tuning** to maximize macro-averaged F1.
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

## Evaluation and Outputs

Each script outputs results to the `models/` directory, including:

- **Feature Importance**: Prints the **top 20 most influential features** used by the models.
- **Consolidated Metrics**: Reports F1, Precision, Recall, ROC-AUC, and PR-AUC.
- **Serialized Models**: Saved in `.json` or subdirectories (for multilabel).
- **Metadata**: Stores feature and target names for downstream inference.
- **Visualizations**: Confusion matrices and training curves (saved in `plots/`).

