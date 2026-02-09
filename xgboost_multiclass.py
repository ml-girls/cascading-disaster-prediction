"""
XGBoost Multiclass Classifier for Cascade Prediction
Problem: "Given a primary event, what is the MOST LIKELY secondary event?"
Assumes each primary triggers exactly ONE secondary (ignores multi-cascade events).

"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import argparse
from xgboost import XGBClassifier

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Configuration for XGBoost multiclass."""
    
    # Paths
    DATA_DIR = Path('random_prepared_data')
    OUTPUT_DIR = Path('models/xgboost_multiclass')
    
    # XGBoost parameters for multiclass
    XGBOOST_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',  # ← MULTICLASS objective!
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Data conversion strategy
    MULTI_CASCADE_STRATEGY = 'most_frequent'  # 'most_frequent', 'random', 'first'


def load_and_convert_data(data_dir: Path, strategy='most_frequent'):
    """
    Load multilabel data and convert to multiclass.
    
    Strategies for handling events with multiple cascades:
    - 'most_frequent': Choose the most common cascade type globally
    - 'random': Randomly select one cascade
    - 'first': Take the first cascade (by column order)
    """
    
    # Load multilabel data
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train_multilabel = np.load(data_dir / 'y_train.npy')
    y_test_multilabel = np.load(data_dir / 'y_test.npy')
    
    # Load metadata
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']
    
    print(f"\nOriginal data (multilabel):")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train_multilabel.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test_multilabel.shape}")
    
    train_cascades_per_event = y_train_multilabel.sum(axis=1)
    test_cascades_per_event = y_test_multilabel.sum(axis=1)
    
    n_multi_train = (train_cascades_per_event > 1).sum()
    n_multi_test = (test_cascades_per_event > 1).sum()
    
    print(f"Train: {n_multi_train:,} / {len(train_cascades_per_event):,} ({n_multi_train/len(train_cascades_per_event)*100:.1f}%)")
    print(f"Test:  {n_multi_test:,} / {len(test_cascades_per_event):,} ({n_multi_test/len(test_cascades_per_event)*100:.1f}%)")
    
    # Convert to multiclass
    y_train_multiclass = convert_multilabel_to_multiclass(
        y_train_multilabel, target_names, strategy
    )
    y_test_multiclass = convert_multilabel_to_multiclass(
        y_test_multilabel, target_names, strategy
    )
    
    print(f"\nConverted data (multiclass):")
    print(f"  y_train: {y_train_multiclass.shape}")
    print(f"  y_test:  {y_test_multiclass.shape}")
    print(f"  Classes: {len(np.unique(y_train_multiclass))}")
    
    # Class distribution
    print(f"\nClass distribution (train, top 10):")
    unique, counts = np.unique(y_train_multiclass, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    for idx in sorted_indices[:10]:
        class_idx = unique[idx]
        count = counts[idx]
        pct = count / len(y_train_multiclass) * 100
        print(f"  {target_names[class_idx]:30s}: {count:6,} ({pct:5.2f}%)")
    
    return (X_train, X_test, y_train_multiclass, y_test_multiclass, 
            feature_names, target_names)


def convert_multilabel_to_multiclass(y_multilabel, target_names, strategy='most_frequent'):
    """
    Convert multilabel matrix to multiclass labels.
    
    Args:
        y_multilabel: (n_samples, n_labels) binary matrix
        target_names: List of label names
        strategy: How to handle multi-cascade events
        
    Returns:
        y_multiclass: (n_samples,) class indices
    """
    
    n_samples = y_multilabel.shape[0]
    y_multiclass = np.zeros(n_samples, dtype=int)
    
    if strategy == 'most_frequent':
        # Choose the most common cascade type globally
        label_frequencies = y_multilabel.sum(axis=0)
        
        for i in range(n_samples):
            active_labels = np.where(y_multilabel[i] == 1)[0]
            
            if len(active_labels) == 0:
                # Should not happen if data is filtered
                y_multiclass[i] = 0
            elif len(active_labels) == 1:
                # Single cascade - easy
                y_multiclass[i] = active_labels[0]
            else:
                # Multiple cascades - choose most frequent globally
                frequencies = label_frequencies[active_labels]
                most_frequent_idx = active_labels[np.argmax(frequencies)]
                y_multiclass[i] = most_frequent_idx
    
    elif strategy == 'random':
        # Randomly select one cascade
        np.random.seed(42)
        
        for i in range(n_samples):
            active_labels = np.where(y_multilabel[i] == 1)[0]
            
            if len(active_labels) == 0:
                y_multiclass[i] = 0
            else:
                y_multiclass[i] = np.random.choice(active_labels)
    
    elif strategy == 'first':
        # Take the first cascade (by column order)
        for i in range(n_samples):
            active_labels = np.where(y_multilabel[i] == 1)[0]
            
            if len(active_labels) == 0:
                y_multiclass[i] = 0
            else:
                y_multiclass[i] = active_labels[0]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return y_multiclass


def train_multiclass_classifier(X_train, y_train, n_classes, config):
    """Train multiclass XGBoost classifier."""
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    
    print(f"\nClass distribution:")
    print(f"  Number of classes: {n_classes}")
    print(f"  Most common class: {counts.max():,} samples")
    print(f"  Least common class: {counts.min():,} samples")
    print(f"  Imbalance ratio: {counts.max() / counts.min():.1f}:1")
    
    # Create model
    params = config.XGBOOST_PARAMS.copy()
    params['num_class'] = n_classes  # Required for multiclass
    
    print(f"\nXGBoost parameters:")
    print(f"  Objective: {params['objective']}")
    print(f"  Num classes: {params['num_class']}")
    print(f"  Max depth: {params['max_depth']}")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  N estimators: {params['n_estimators']}")
    
    model = XGBClassifier(**params)
    
    import time
    start_time = time.time()
    
    model.fit(X_train, y_train, verbose=False)
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    return model

def predict(model, X_test):
    """Generate predictions."""
    
    # Predict class labels
    y_pred = model.predict(X_test)
    
    # Predict probabilities
    y_prob = model.predict_proba(X_test)
    
    print(f"\n✓ Predictions generated:")
    print(f"  y_pred: {y_pred.shape}")
    print(f"  y_prob: {y_prob.shape}")
    
    # Predicted distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"\nPredicted class distribution:")
    print(f"  Unique classes predicted: {len(unique)}")
    print(f"  Most predicted: {counts.max():,} times")
    
    return y_pred, y_prob

def evaluate_multiclass(y_test, y_pred, y_prob, target_names):
    """Comprehensive multiclass evaluation."""
    
    results = {}
    
    # Overall metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    
    results['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    results['f1_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
    results['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    results['precision_micro'] = precision_score(y_test, y_pred, average='micro', zero_division=0)
    results['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    results['recall_micro'] = recall_score(y_test, y_pred, average='micro', zero_division=0)
    results['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:             {results['accuracy']:.4f}")
    print(f"\nF1 (weighted):        {results['f1_weighted']:.4f}  ← PRIMARY METRIC")
    print(f"F1 (macro):           {results['f1_macro']:.4f}")
    print(f"F1 (micro):           {results['f1_micro']:.4f}")
    
    print(f"\nPrecision (weighted): {results['precision_weighted']:.4f}")
    print(f"Precision (macro):    {results['precision_macro']:.4f}")
    
    print(f"\nRecall (weighted):    {results['recall_weighted']:.4f}")
    print(f"Recall (macro):       {results['recall_macro']:.4f}")
    
    # Classification report
    print("\n--- Classification Report (Top 10 Classes) ---")
    report = classification_report(
        y_test, y_pred, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Extract per-class metrics
    per_class_metrics = []
    for i, name in enumerate(target_names):
        if name in report:
            per_class_metrics.append({
                'class': name,
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': int(report[name]['support'])
            })
    
    # Sort by support and show top 10
    per_class_metrics_sorted = sorted(per_class_metrics, key=lambda x: x['support'], reverse=True)
    
    print(f"{'Class':<30} {'Support':>8} {'F1':>6} {'Prec':>6} {'Recall':>6}")
    print("-" * 75)
    
    for m in per_class_metrics_sorted[:10]:
        print(f"{m['class']:<30} {m['support']:>8,} {m['f1']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f}")
    
    if len(per_class_metrics_sorted) > 10:
        print(f"... ({len(per_class_metrics_sorted) - 10} more classes)")
    
    results['per_class_metrics'] = per_class_metrics
    
    return results

def plot_results(y_test, y_pred, results, target_names, output_dir):
    """Create visualization plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Confusion Matrix (top 10 classes only)
    # Get top 10 most common classes
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    top_10_indices = unique_test[np.argsort(-counts_test)[:10]]
    
    # Filter data to top 10
    mask = np.isin(y_test, top_10_indices) & np.isin(y_pred, top_10_indices)
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]
    
    top_10_names = [target_names[i] for i in top_10_indices]
    
    cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=top_10_indices)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=top_10_names,
        yticklabels=top_10_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Top 10 Cascade Types')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix")
    plt.close()
    
    # Plot 2: Per-class F1 scores
    per_class = results['per_class_metrics']
    df = pd.DataFrame(per_class)
    df = df.sort_values('f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
    
    bars = ax.barh(df['class'], df['f1'])
    
    for i, bar in enumerate(bars):
        if df['f1'].iloc[i] >= 0.5:
            bar.set_color('forestgreen')
        elif df['f1'].iloc[i] >= 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('crimson')
    
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-Class F1 Scores (XGBoost Multiclass)')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_f1.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_model(model, results, feature_names, target_names, output_dir):
    """Save model and results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'multiclass_model.json'
    model.save_model(model_path)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'target_names': target_names,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'xgboost_params': Config.XGBOOST_PARAMS,
            'multi_cascade_strategy': Config.MULTI_CASCADE_STRATEGY
        },
        'results': results
    }
    
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save text report
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("XGBoost Multiclass Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:             {results['accuracy']:.4f}\n")
        f.write(f"  F1 (weighted):        {results['f1_weighted']:.4f}\n")
        f.write(f"  F1 (macro):           {results['f1_macro']:.4f}\n")
        f.write(f"  Precision (weighted): {results['precision_weighted']:.4f}\n")
        f.write(f"  Recall (weighted):    {results['recall_weighted']:.4f}\n\n")
        
        f.write("Per-Class Results:\n")
        f.write(f"{'Class':<30} {'Support':>8} {'F1':>6} {'Prec':>6} {'Recall':>6}\n")
        f.write("-"*80 + "\n")
        
        sorted_classes = sorted(results['per_class_metrics'], 
                               key=lambda x: x['support'], reverse=True)
        for m in sorted_classes:
            f.write(f"{m['class']:<30} {m['support']:>8,} {m['f1']:>6.3f} "
                   f"{m['precision']:>6.3f} {m['recall']:>6.3f}\n")
    
    print(f"\nOutput directory: {output_dir.absolute()}")


def main(data_dir):
    """Run multiclass XGBoost pipeline."""  
    
    config = Config()
    
    # Load and convert data
    X_train, X_test, y_train, y_test, feature_names, target_names = \
        load_and_convert_data(data_dir, config.MULTI_CASCADE_STRATEGY)
    
    # Train
    n_classes = len(target_names)
    model = train_multiclass_classifier(X_train, y_train, n_classes, config)
    
    # Predict
    y_pred, y_prob = predict(model, X_test)
    
    # Evaluate
    results = evaluate_multiclass(y_test, y_pred, y_prob, target_names)
    
    # Visualize
    plot_results(y_test, y_pred, results, target_names, config.OUTPUT_DIR)
    
    # Save
    save_model(model, results, feature_names, target_names, config.OUTPUT_DIR)
    
    # Summary
    print(f"\nKey Results:")
    print(f"  Accuracy:             {results['accuracy']:.4f}")
    print(f"  F1 (weighted):        {results['f1_weighted']:.4f}")
    print(f"  F1 (macro):           {results['f1_macro']:.4f}")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {results['recall_weighted']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Cascade Detection')
    parser.add_argument('--data_dir', type=str, help='Path to prepared data directory')
    args = parser.parse_args()
    main(args.data_dir)