"""
Event Labeling Module

Creates multilabel targets for cascade prediction.

Problem: "Given event X, what types of secondary events occur (if any)?"
Labels: Set of event types like {Flood, Tornado} or empty set {}
"""

import pandas as pd
import numpy as np
from typing import List, Set
from collections import defaultdict

from .cascade_identification import CascadePair


def create_cascade_labels(
    df: pd.DataFrame,
    cascades: List[CascadePair]
) -> pd.DataFrame:
    """
    Create multilabel cascade targets for each event.
    
    For each event, the target is the SET of secondary event types it triggers.
    Events that don't trigger cascades have an empty set.
    
    Args:
        df: DataFrame with EVENT_ID column
        cascades: List of CascadePair objects from identify_cascades()
        
    Returns:
        DataFrame with added columns:
        - target: List of secondary event types (empty list if no cascade)
        - is_cascade_result: True if this event was triggered by another
        
    Example:
        >>> cascades = identify_cascades(df, config)
        >>> df_labeled = create_cascade_labels(df, cascades)
        >>> df_labeled[df_labeled['target'].apply(len) > 0]['target'].head()
        42     [Flash Flood, Flood]
        156    [Tornado]
        203    [Flash Flood]
    """
    df = df.copy()
    
    # Initialize columns
    df['target'] = [[] for _ in range(len(df))]  # Multilabel: list of secondary types
    df['is_cascade_result'] = False
    
    if not cascades:
        return df
    
    # Build lookup: primary_event_id -> set of secondary event types
    primary_to_secondary_types: dict[int, Set[str]] = defaultdict(set)
    secondary_event_ids: set = set()
    
    for cascade in cascades:
        primary_to_secondary_types[cascade.primary_event_id].add(cascade.secondary_event_type)
        secondary_event_ids.add(cascade.secondary_event_id)
    
    # Apply multilabel targets
    event_id_to_idx = {eid: idx for idx, eid in enumerate(df['EVENT_ID'])}
    for event_id, secondary_types in primary_to_secondary_types.items():
        if event_id in event_id_to_idx:
            idx = event_id_to_idx[event_id]
            df.at[df.index[idx], 'target'] = sorted(list(secondary_types))
    
    # Mark cascade results
    df['is_cascade_result'] = df['EVENT_ID'].isin(secondary_event_ids)
    
    return df


def get_all_classes(df: pd.DataFrame) -> List[str]:
    """Get sorted list of all unique secondary event types."""
    all_types = set()
    for labels in df['target']:
        all_types.update(labels)
    return sorted(list(all_types))


def to_binary_matrix(df: pd.DataFrame, classes: List[str] = None) -> pd.DataFrame:
    """
    Convert multilabel targets to binary matrix for sklearn.
    
    Args:
        df: DataFrame with 'target' column (list of labels)
        classes: List of class names. If None, inferred from data.
        
    Returns:
        DataFrame with binary columns for each class
    """
    if classes is None:
        classes = get_all_classes(df)
    
    # Create binary matrix
    binary_data = []
    for labels in df['target']:
        row = {cls: 1 if cls in labels else 0 for cls in classes}
        binary_data.append(row)
    
    return pd.DataFrame(binary_data, index=df.index)


def get_label_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics for the labeled dataset."""
    
    n_total = len(df)
    n_with_cascade = df['target'].apply(lambda x: len(x) > 0).sum()
    all_classes = get_all_classes(df)
    
    # Count per class
    class_counts = defaultdict(int)
    for labels in df['target']:
        for label in labels:
            class_counts[label] += 1
    
    return {
        'total_events': n_total,
        'events_with_cascade': n_with_cascade,
        'cascade_rate': f"{n_with_cascade / n_total * 100:.2f}%",
        'num_classes': len(all_classes),
        'class_counts': dict(sorted(class_counts.items(), key=lambda x: -x[1]))
    }
