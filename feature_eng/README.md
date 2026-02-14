# Feature Engineering Pipeline

This sub-package handles the transformation of raw labeled data into feature matrices suitable for machine learning models. It supports both base event-level feature engineering and advanced aggregate statistics.

## Data Preparation (`prepare_data.py`)

Handles the splitting, filtering, and transformation workflow.

### Usage

```bash
python prepare_data.py --split_type [random|chronological] --filter_cascades [True|False]
```

### Arguments

- `--split_type`: 
    - `chronological`: Splits data based on time (train on past, test on future). Recommended for evaluating temporal generalization.
    - `random`: Randomized split with stratification based on the presence of any cascade.
- `--filter_cascades`:
    - `True`: Filters the dataset to ONLY include events that triggered at least one cascade. (Useful for conditional cascade-type prediction).
    - `False`: Keeps all events. (Required for binary "is_cascade" detection).

## Feature Engineering Components

### Base Features (`features.py`)

The pipeline generates several categories of features. Now includes improved impact parsing and detailed event metrics.

| Category | Description |
| :--- | :--- |
| **Temporal** | Day of week, month, hour, and season of the event. |
| **Impact** | Parses damage strings (e.g., '10.00K', '5.00M') into numeric USD (`total_damage_usd`). Includes `log_damage`, injuries, deaths, and a `severity_score` (financial + human impact). |
| **Event Details** | Magnitude (wind speed/hail size), CZ Type (Marine/Zone), and Flood Cause (Ice Jam/Rain/Snowmelt). |
| **Spatial** | Latitude, longitude, elevation, and calculated `event_path_length_miles` (Haversine distance). |
| **Tornado** | Specific metrics: intensity (F-scale), path length, width, and path area. |
| **Historical** | Moving window statistics (7 and 30 days) for cumulative damage and event frequency in the same location/state. |

### Aggregate Statistics (`aggregate_features.py`)

The `AggregateFeatureTransformer` "learns" historical patterns from the training set to create predictive features:

1.  **Conditional Probabilities**: Calculates $P(\text{Secondary} | \text{Primary})$ — the historical likelihood of a specific secondary event occurring given a primary event type.
2.  **Location/State/Type Statistics**: 
    - **Cascade Rate**: Success rate of events triggering cascades in specific `LOCATION_KEY` or `STATE`.
    - **Average Damage**: Mean historical damage for the location, state, or event type.
    - **Average Severity**: Mean human/financial impact score for the location, state, or event type.

## Output Structure

Processed data is saved to subdirectories based on configuration (e.g., `random_data/`, `chronological_filtered_data/`):

- `X_train.npy`, `X_test.npy`: Feature matrices.
- `y_train.npy`, `y_test.npy`: Target labels (multilabel format).
- `metadata.pkl`: Stores `feature_names`, `target_names`, and the configuration flags.


