import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Optional
from features import engineer_base_features, get_feature_columns
from aggregate_features import AggregateFeatureTransformer

SPLIT_TYPE = 'chronological'
DATA_DIR = Path(__file__).parent.parent.parent / "labeled_data"
OUTPUT_DIR = Path(__file__).parent.parent / f"{SPLIT_TYPE}_prepared_data"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(year_range=(2010, 2025)):
    """Load and filter to cascade-related events only."""

    events_path = DATA_DIR / "events_labeled.csv"
    labels_path = DATA_DIR / "labels_binary.csv"

    events_df = pd.read_csv(events_path, low_memory=False)
    labels_binary = pd.read_csv(labels_path)
    
    if 'EVENT_ID' in labels_binary.columns:
        label_cols = [c for c in labels_binary.columns if c != 'EVENT_ID']
        
        combined = events_df.merge(
            labels_binary,
            on='EVENT_ID',
            how='inner',
            validate='one_to_one'  # Ensures 1:1 relationship
        )
        
        # Split back into events and labels
        labels_binary = combined[label_cols]
        events_df = combined.drop(columns=label_cols)
    
    # Parse dates
    for col in ['BEGIN_DATETIME', 'END_DATETIME']:
        if col in events_df.columns:
            events_df[col] = pd.to_datetime(events_df[col], errors='coerce')
    
    year_mask = (events_df['BEGIN_DATETIME'].dt.year >= year_range[0]) & \
                (events_df['BEGIN_DATETIME'].dt.year <= year_range[1])
    
    events_df = events_df[year_mask].reset_index(drop=True)
    labels_binary = labels_binary[year_mask].reset_index(drop=True)
    
    print(f"After year filter ({year_range[0]}-{year_range[1]}): {len(events_df):,} events")
    has_any_cascade = labels_binary.sum(axis=1) > 0
    
    print(f"\nCascade filtering:")
    print(f"  Events with cascades: {has_any_cascade.sum():,}")
    print(f"  Events without cascades: {(~has_any_cascade).sum():,}")
    
    events_df = events_df[has_any_cascade].reset_index(drop=True)
    labels_binary = labels_binary[has_any_cascade].reset_index(drop=True)
    
    n_events = len(events_df)
    n_positive_labels = labels_binary.values.sum()
    n_total_labels = labels_binary.size
    
    # Per-event statistics
    labels_per_event = labels_binary.sum(axis=1)
    avg_labels_per_event = labels_per_event.mean()
    max_labels_per_event = labels_per_event.max()
    
    # Per-label statistics
    events_per_label = labels_binary.sum(axis=0)
    avg_events_per_label = events_per_label.mean()
    
    print(f"  Total events: {n_events:,}")
    print(f"  Total positive labels: {n_positive_labels:,}")
    print(f"  Label sparsity: {n_positive_labels/n_total_labels*100:.2f}%")
    
    print(f"  Avg cascades per event: {avg_labels_per_event:.2f}")
    print(f"  Max cascades per event: {max_labels_per_event}")
    print(f"  Avg events per cascade type: {avg_events_per_label:.1f}")
    
    cascade_dist = labels_per_event.value_counts().sort_index()
    for n_cascades, count in cascade_dist.items():
        pct = count / n_events * 100
        print(f"  {n_cascades} cascades: {count:,} events ({pct:.1f}%)")

    cascade_pairs_path = DATA_DIR / "cascade_pairs.csv"
    
    cascade_pairs = None
    if cascade_pairs_path.exists():
        cascade_pairs = pd.read_csv(cascade_pairs_path)
        if 'primary_begin_time' in cascade_pairs.columns:
            cascade_pairs['primary_begin_time'] = pd.to_datetime(
                cascade_pairs['primary_begin_time'], errors='coerce'
            )
    
    return events_df, cascade_pairs, labels_binary

def split_data(events_df, labels_binary, split_type='chronological', test_size=0.2):
    if split_type == 'chronological':
        sort_idx = events_df['BEGIN_DATETIME'].argsort()
        events_df = events_df.iloc[sort_idx].reset_index(drop=True)
        labels_binary = labels_binary.iloc[sort_idx].reset_index(drop=True)
        split_idx = int(len(events_df) * (1 - test_size))
        return events_df.iloc[:split_idx], events_df.iloc[split_idx:], labels_binary.iloc[:split_idx], labels_binary.iloc[split_idx:]
    elif split_type == 'random':
        from sklearn.model_selection import train_test_split
        any_cascade = (labels_binary.values.sum(axis=1) > 0).astype(int)
        train_idx, test_idx = train_test_split(np.arange(len(events_df)), test_size=test_size, random_state=RANDOM_STATE, stratify=any_cascade)
        return events_df.iloc[train_idx], events_df.iloc[test_idx], labels_binary.iloc[train_idx], labels_binary.iloc[test_idx]
    raise ValueError(f"Unknown split_type: {split_type}")

def engineer_features(train_events, test_events, cascade_pairs, split_type, include_historical=True):
    if split_type == 'random':
        include_historical = False
    
    train_featured = engineer_base_features(train_events, include_historical=include_historical)
    test_featured = engineer_base_features(test_events, include_historical=include_historical)
    
    if split_type == 'chronological' and cascade_pairs is not None:
        agg_transformer = AggregateFeatureTransformer()
        train_featured = agg_transformer.fit_transform(train_featured, cascade_pairs)
        test_featured = agg_transformer.transform(test_featured)
        
    feature_cols = get_feature_columns(train_featured)
    return train_featured, test_featured, feature_cols

def prepare_data(include_historical=True, split_type='chronological'):
    events_df, cascade_pairs, labels_binary = load_data(year_range=(2010, 2025))

    train_events, test_events, train_labels, test_labels = split_data(events_df, labels_binary, split_type=split_type, test_size=TEST_SIZE)
    
    cascade_pairs_train = None
    if cascade_pairs is not None and split_type == 'chronological':
        cutoff_time = train_events['BEGIN_DATETIME'].max()
        cascade_pairs_train = cascade_pairs[cascade_pairs['primary_begin_time'] <= cutoff_time].copy() if 'primary_begin_time' in cascade_pairs.columns else \
                             cascade_pairs[cascade_pairs['primary_event_id'].isin(set(train_events['EVENT_ID']))].copy()
    
    train_featured, test_featured, feature_cols = engineer_features(train_events, test_events, cascade_pairs_train, split_type, include_historical)
    
    X_train, X_test = train_featured[feature_cols].fillna(0), test_featured[feature_cols].fillna(0)
    y_train, y_test = train_labels.values, test_labels.values
    target_names = train_labels.columns.tolist()

    zero_var_mask = (X_train == 0).all(axis=0)
    n_zero_var = zero_var_mask.sum()
    
    if n_zero_var > 0:
        print(f"  Found {n_zero_var} zero-variance features")
        feature_cols = [f for f, keep in zip(feature_cols, ~zero_var_mask) if keep]
        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]
        print(f"  Kept {len(feature_cols)} features")
    
    # Remove labels with < 10 positive samples
    print("\nFiltering labels with insufficient samples...")
    label_counts = y_train.sum(axis=0)
    label_mask = label_counts >= 10
    n_removed = (~label_mask).sum()
    
    if n_removed > 0:
        print(f"  Removing {n_removed} labels with < 10 samples")
        y_train = y_train[:, label_mask]
        y_test = y_test[:, label_mask]
        target_names = [t for t, keep in zip(target_names, label_mask) if keep]
        print(f"  Kept {len(target_names)} labels")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train.values); np.save(OUTPUT_DIR / "X_test.npy", X_test.values)
    np.save(OUTPUT_DIR / "y_train.npy", y_train); np.save(OUTPUT_DIR / "y_test.npy", y_test)
    
    with open(OUTPUT_DIR / "metadata.pkl", "wb") as f:
        pickle.dump({'feature_names': feature_cols, 'target_names': target_names, 'split_type': split_type}, f)
        
    return X_train, X_test, y_train, y_test, feature_cols, target_names

if __name__ == "__main__":
    if SPLIT_TYPE == 'random':
        prepare_data(include_historical=False, split_type=SPLIT_TYPE)
    elif SPLIT_TYPE == 'chronological':
        prepare_data(include_historical=True, split_type=SPLIT_TYPE)
    else:
        raise ValueError(f"Unknown split_type: {SPLIT_TYPE}")