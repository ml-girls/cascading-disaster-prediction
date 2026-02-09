"""
XGBoost Baseline for Multilabel Cascade Prediction
Binary Relevance Method
Per-label scale_pos_weight for class imbalance
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from xgboost import XGBClassifier
from joblib import Parallel, delayed

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, hamming_loss, jaccard_score
)

import matplotlib.pyplot as plt

class Config:
    """Configuration for XGBoost baseline."""
    
    # Paths
    DATA_DIR = Path('random_prepared_data')
    OUTPUT_DIR = Path('models/xgboost_baseline')
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Class imbalance
    AUTO_SCALE_POS_WEIGHT = True
    MIN_POSITIVE_SAMPLES = 10
    
    # Training
    PARALLEL_TRAINING = True
    N_JOBS = -1
    
    # Evaluation
    DECISION_THRESHOLD = 0.5

def load_prepared_data(data_dir: Path):
    """Load prepared data from numpy files."""
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Load metadata
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    print(f"\nMetadata:")
    print(f"  Features: {len(feature_names)}")
    print(f"  Targets: {len(target_names)}")
    print(f"  Split type: {metadata.get('split_type', 'unknown')}")
    
    # Data quality
    print(f"\nData quality:")
    has_nan = np.isnan(X_train).any() or np.isnan(X_test).any()
    has_inf = np.isinf(X_train).any() or np.isinf(X_test).any()
    
    # Label distribution
    train_rate = y_train.sum() / y_train.size * 100
    test_rate = y_test.sum() / y_test.size * 100
    print(f"\nLabel distribution:")
    print(f"  Train cascade rate: {train_rate:.2f}%")
    print(f"  Test cascade rate:  {test_rate:.2f}%")
    
    return X_train, X_test, y_train, y_test, feature_names, target_names


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for imbalanced binary labels."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos if pos > 0 else 1.0


def train_single_label(X_train, y_train, label_name, config):
    """Train XGBoost for a single label."""
    
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    
    # Skip labels with too few positive samples
    if n_pos < config.MIN_POSITIVE_SAMPLES:
        print(f"{label_name:30s} Skipped (only {n_pos} positive samples)")
        return None
    
    # Compute per-label scale_pos_weight
    scale = compute_scale_pos_weight(y_train) if config.AUTO_SCALE_POS_WEIGHT else 1.0
    
    print(f"Training {label_name:30s} | Pos: {n_pos:6,} | Neg: {n_neg:6,} | Scale: {scale:6.2f}")
    
    # Create and train model
    params = config.XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    return model


def train_all_labels(X_train, y_train, target_names, config):
    """Train one XGBoost model per label (Binary Relevance)."""
    
    n_labels = y_train.shape[1]
    
    if config.PARALLEL_TRAINING:
        print(f"Parallel training: Enabled ({config.N_JOBS} cores)\n")
        
        models = Parallel(n_jobs=config.N_JOBS, verbose=0)(
            delayed(train_single_label)(X_train, y_train[:, i], target_names[i], config)
            for i in range(n_labels)
        )
    else:
        print("Parallel training: Disabled\n")
        
        models = []
        for i in range(n_labels):
            model = train_single_label(X_train, y_train[:, i], target_names[i], config)
            models.append(model)
    
    n_trained = sum(1 for m in models if m is not None)
    n_skipped = n_labels - n_trained
    
    print(f"\nSuccessfully trained {n_trained}/{n_labels} models")
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} labels (insufficient positive samples)")
    
    return models

def predict_multilabel(models, X_test, threshold=0.5):
    """Predict using all trained models."""
    
    n_samples = X_test.shape[0]
    n_labels = len(models)
    
    y_pred = np.zeros((n_samples, n_labels), dtype=int)
    y_prob = np.zeros((n_samples, n_labels), dtype=float)
    
    for i, model in enumerate(models):
        if model is None:
            continue
        
        # Get probabilities
        probs = model.predict_proba(X_test)[:, 1]
        y_prob[:, i] = probs
        
        # Apply threshold
        y_pred[:, i] = (probs >= threshold).astype(int)
    
    return y_pred, y_prob

def evaluate_multilabel(y_true, y_pred, y_prob, target_names):
    """Comprehensive multilabel evaluation."""
    
    results = {}
    
    # Overall metrics
    results['hamming_loss'] = hamming_loss(y_true, y_pred)
    results['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    results['jaccard_samples'] = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    results['subset_accuracy'] = (y_true == y_pred).all(axis=1).mean()
    
    print("\n--- Overall Metrics ---")
    print(f"Hamming Loss:         {results['hamming_loss']:.4f}")
    print(f"Jaccard (macro):      {results['jaccard_macro']:.4f}")
    print(f"Jaccard (samples):    {results['jaccard_samples']:.4f}")
    print(f"\nF1 (macro):           {results['f1_macro']:.4f}")
    print(f"F1 (micro):           {results['f1_micro']:.4f}")
    print(f"F1 (weighted):        {results['f1_weighted']:.4f}")
    print(f"\nPrecision (macro):    {results['precision_macro']:.4f}")
    print(f"Precision (micro):    {results['precision_micro']:.4f}")
    print(f"\nRecall (macro):       {results['recall_macro']:.4f}")
    print(f"Recall (micro):       {results['recall_micro']:.4f}")
    print(f"\nSubset Accuracy:      {results['subset_accuracy']:.4f}")
    
    # Per-label metrics
    per_label_metrics = []
    
    print("\n--- Per-Label Metrics ---")
    print(f"{'Label':<30} {'Support':>8} {'F1':>6} {'Prec':>6} {'Recall':>6} {'PR-AUC':>7}")
    print("-" * 80)
    
    for i, label_name in enumerate(target_names):
        support = y_true[:, i].sum()
        
        if support == 0:
            metrics = {
                'label': label_name,
                'support': 0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'pr_auc': 0.0
            }
        else:
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            
            try:
                pr_auc = average_precision_score(y_true[:, i], y_prob[:, i])
            except:
                pr_auc = 0.0
            
            metrics = {
                'label': label_name,
                'support': int(support),
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'pr_auc': pr_auc
            }
        
        per_label_metrics.append(metrics)
        
        print(f"{label_name:<30} {metrics['support']:>8,} {metrics['f1']:>6.3f} "
              f"{metrics['precision']:>6.3f} {metrics['recall']:>6.3f} {metrics['pr_auc']:>7.3f}")
    
    results['per_label_metrics'] = per_label_metrics
    
    return results


def plot_results(results, output_dir):
    """Create visualization plots."""
    
    per_label = results['per_label_metrics']
    df = pd.DataFrame(per_label)
    df = df[df['support'] > 0].copy()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: F1 Scores
    df_sorted = df.sort_values('f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
    bars = ax.barh(df_sorted['label'], df_sorted['f1'])
    
    for i, bar in enumerate(bars):
        if df_sorted['f1'].iloc[i] >= 0.5:
            bar.set_color('forestgreen')
        elif df_sorted['f1'].iloc[i] >= 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('crimson')
    
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-Label F1 Scores (XGBoost Binary Relevance)')
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_label_f1.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved F1 plot")
    plt.close()
    
    # Plot 2: Precision vs Recall
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        df['recall'], 
        df['precision'],
        s=df['support'] / df['support'].max() * 500,
        c=df['f1'],
        cmap='RdYlGn',
        alpha=0.6,
        edgecolors='black'
    )
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (Per Label)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_models(models, target_names, feature_names, results, output_dir):
    """Save models and results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for i, (model, label_name) in enumerate(zip(models, target_names)):
        if model is not None:
            clean_name = label_name.replace('/', '_').replace(' ', '_')
            model_path = output_dir / f'model_{i:02d}_{clean_name}.json'
            model.save_model(model_path)
    
    # Save metadata
    metadata = {
        'target_names': target_names,
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'xgboost_params': Config.XGBOOST_PARAMS,
            'auto_scale_pos_weight': Config.AUTO_SCALE_POS_WEIGHT,
            'decision_threshold': Config.DECISION_THRESHOLD
        },
        'results': results
    }
    
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save text report
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("XGBoost Binary Relevance Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        
        f.write("Overall Metrics:\n")
        for key, value in results.items():
            if key != 'per_label_metrics':
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nPer-Label Results:\n")
        f.write(f"{'Label':<30} {'Support':>8} {'F1':>6} {'Prec':>6} {'Recall':>6}\n")
        f.write("-"*80 + "\n")
        for m in results['per_label_metrics']:
            f.write(f"{m['label']:<30} {m['support']:>8,} {m['f1']:>6.3f} "
                   f"{m['precision']:>6.3f} {m['recall']:>6.3f}\n")
    
    n_saved = sum(1 for m in models if m is not None)
    print(f"\nSaved {n_saved} models")
    print(f"Saved metadata.pkl")
    print(f"Saved results.txt")

def main(data_dir):
    """Run XGBoost Binary Relevance baseline."""
    
    config = Config()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names = \
        load_prepared_data(data_dir)
    
    # Train models
    models = train_all_labels(X_train, y_train, target_names, config)
    
    # Predict
    y_pred, y_prob = predict_multilabel(models, X_test, config.DECISION_THRESHOLD)
    
    print(f"\nPredictions: {y_pred.shape}")
    print(f"Probabilities: {y_prob.shape}")
    
    # Evaluate
    results = evaluate_multilabel(y_test, y_pred, y_prob, target_names)
    
    # Visualize
    plot_results(results, config.OUTPUT_DIR)
    
    # Save
    save_models(models, target_names, feature_names, results, config.OUTPUT_DIR)
    
    # Summary
    print(f"\nKey Results:")
    print(f"  F1 (macro):           {results['f1_macro']:.4f}")
    print(f"  F1 (micro):           {results['f1_micro']:.4f}")
    print(f"  F1 (weighted):        {results['f1_weighted']:.4f}")
    print(f"  Precision (macro):    {results['precision_macro']:.4f}")
    print(f"  Recall (macro):       {results['recall_macro']:.4f}")
    print(f"  Hamming Loss:         {results['hamming_loss']:.4f}")
    print(f"  Subset Accuracy:      {results['subset_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Cascade Detection')
    parser.add_argument('--data_dir', type=str, help='Path to prepared data directory')
    args = parser.parse_args()
    main(args.data_dir)