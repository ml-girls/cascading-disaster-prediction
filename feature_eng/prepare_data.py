import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Optional
import argparse
import ast
from datetime import datetime
from features import engineer_base_features, get_feature_columns
from aggregate_features import AggregateFeatureTransformer

# Path to the primary labeled dataset
FULL_DATA_PATH = Path("/Users/varshel/Documents/10718/Project/cascading-disaster-prediction/data/processed/events_labeled_full.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2

CASCADE_TYPES = [
    'Avalanche', 'Blizzard', 'Coastal Flood', 'Cold/Wind Chill', 'Debris Flow',
    'Dense Fog', 'Dust Storm', 'Excessive Heat', 'Extreme Cold/Wind Chill',
    'Flash Flood', 'Flood', 'Frost/Freeze', 'Hail', 'Heat', 'Heavy Rain',
    'Heavy Snow', 'High Surf', 'High Wind', 'Ice Storm', 'Lightning',
    'Marine Hail', 'Marine Thunderstorm Wind', 'Rip Current', 'Storm Surge/Tide',
    'Thunderstorm Wind', 'Tornado', 'Waterspout', 'Wildfire', 'Winter Weather'
]

def load_data(year_range=(2010, 2025), filter_cascades=True):
    """Load unified dataset and expand label targets."""

    events_path = FULL_DATA_PATH

    print(f"Loading data from: {events_path}")
    df = pd.read_csv(events_path, low_memory=False)
    
    # Pre-process columns
    if 'is_cascade_result' in df.columns:
        df['is_cascade_result'] = df['is_cascade_result'].astype(str).str.lower().map({
            'true': 1, 'false': 0, '1': 1, '0': 0
        }).fillna(0).astype(int)    
    
    # Expand labels from 'target' column
    def parse_target(t):
        if pd.isna(t) or t == '[]': return []
        try:
            if isinstance(t, str) and t.startswith('['):
                return ast.literal_eval(t)
            return []
        except: return []

    print("Expanding labels...")
    target_series = df['target'].apply(parse_target)
    labels_binary = pd.DataFrame(0, index=df.index, columns=CASCADE_TYPES)
    df['triggered_any_cascade'] = (labels_binary.sum(axis=1) > 0).astype(int)
    events_df = df.copy()
    
    for label in CASCADE_TYPES:
        labels_binary[label] = target_series.apply(lambda x: 1 if label in x else 0)
    
    # Parse dates
    for col in ['BEGIN_DATETIME', 'END_DATETIME']:
        if col in events_df.columns:
            events_df[col] = pd.to_datetime(events_df[col], errors='coerce')
    
    # Filter by year
    year_mask = (events_df['BEGIN_DATETIME'].dt.year >= year_range[0]) & \
                (events_df['BEGIN_DATETIME'].dt.year <= year_range[1])
    
    events_df = events_df[year_mask].reset_index(drop=True)
    labels_binary = labels_binary[year_mask].reset_index(drop=True)
    
    print(f"Total events in range: {len(events_df):,}")
    
    if filter_cascades:
        has_any_cascade = labels_binary.sum(axis=1) > 0
        print(f"Filtering to {has_any_cascade.sum():,} ({has_any_cascade.sum()/len(events_df)*100:.1f}%) cascade events...")
        events_df = events_df[has_any_cascade].reset_index(drop=True)
        labels_binary = labels_binary[has_any_cascade].reset_index(drop=True)
    
    return events_df, labels_binary

def split_data(events_df, labels_binary, split_type='chronological', test_size=0.2):
    if split_type == 'chronological':
        sort_idx = events_df['BEGIN_DATETIME'].argsort()
        events_df = events_df.iloc[sort_idx].reset_index(drop=True)
        labels_binary = labels_binary.iloc[sort_idx].reset_index(drop=True)
        split_idx = int(len(events_df) * (1 - test_size))
        return events_df.iloc[:split_idx], events_df.iloc[split_idx:], labels_binary.iloc[:split_idx], labels_binary.iloc[split_idx:]
    if split_type == 'random':
        from sklearn.model_selection import train_test_split
        any_cascade = (labels_binary.values.sum(axis=1) > 0).astype(int)
        train_idx, test_idx = train_test_split(np.arange(len(events_df)), test_size=test_size, random_state=RANDOM_STATE, stratify=any_cascade)
        return events_df.iloc[train_idx], events_df.iloc[test_idx], labels_binary.iloc[train_idx], labels_binary.iloc[test_idx]

def engineer_features(train_events, test_events, train_labels, split_type, include_historical=True):
    # Disable historical features for random split to prevent leakage
    if split_type == 'random':
        include_historical = False
    
    # Base features
    train_featured = engineer_base_features(train_events, include_historical=include_historical)
    test_featured = engineer_base_features(test_events, include_historical=include_historical)
    
    # Aggregate features (Location stats + Transition probabilities)
    # Only meaningful for chronological split where we fit on past and transform present
    if split_type == 'chronological':
        agg_transformer = AggregateFeatureTransformer()
        train_featured = agg_transformer.fit_transform(train_featured, train_labels)
        test_featured = agg_transformer.transform(test_featured)
        
    feature_cols = get_feature_columns(train_featured)
    return train_featured, test_featured, feature_cols

def prepare_data(include_historical=True, split_type='chronological', filter_cascades=True):
    suffix = "_filtered" if filter_cascades else ""
    output_dir = Path(__file__).parent.parent / f"{split_type}{suffix}_prepared_data"

    events_df, labels_binary = load_data(year_range=(2010, 2025), filter_cascades=filter_cascades)
    train_events, test_events, train_labels, test_labels = split_data(events_df, labels_binary, split_type=split_type, test_size=TEST_SIZE)
    
    train_featured, test_featured, feature_cols = engineer_features(
        train_events, test_events, train_labels, split_type, include_historical
    )
    
    X_train, X_test = train_featured[feature_cols].fillna(0), test_featured[feature_cols].fillna(0)
    y_train, y_test = train_labels.values, test_labels.values
    target_names = train_labels.columns.tolist()

    # Remove zero-variance features
    zero_var_mask = (X_train == 0).all(axis=0) | (X_train.std(axis=0) == 0)
    if zero_var_mask.any():
        print(f"  Removing {zero_var_mask.sum()} zero-variance features...")
        feature_cols = [f for f, keep in zip(feature_cols, ~zero_var_mask) if keep]
        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]
    
    # Remove labels with < 10 positive samples in train
    print("\nFiltering labels with insufficient samples (< 10)...")
    label_counts = y_train.sum(axis=0)
    label_mask = label_counts >= 10
    if (~label_mask).any():
        print(f"  Removing {(~label_mask).sum()} labels. Kept {label_mask.sum()} labels.")
        y_train = y_train[:, label_mask]
        y_test = y_test[:, label_mask]
        target_names = [t for t, keep in zip(target_names, label_mask) if keep]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train.values)
    np.save(output_dir / "X_test.npy", X_test.values)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    
    metadata = {
        'feature_names': feature_cols,
        'target_names': target_names,
        'split_type': split_type,
        'filtered': filter_cascades,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"\n✓ Preparation complete. Saved to {output_dir}")
    print(f"  Features: {len(feature_cols)}, Labels: {len(target_names)}")
    return X_train, X_test, y_train, y_test, feature_cols, target_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for cascade detection')
    parser.add_argument('--split_type', type=str, default='chronological', choices=['random', 'chronological'], help='Split type')
    parser.add_argument('--filter_cascades', type=str, default='False', help='Filter to cascades only (True/False)')
    args = parser.parse_args()

    filter_val = args.filter_cascades.lower() == 'true'
    include_hist = (args.split_type == 'chronological')
    
    prepare_data(include_historical=include_hist, split_type=args.split_type, filter_cascades=filter_val)