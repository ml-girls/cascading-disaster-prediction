"""
Simple Binary XGBoost Classifier for Cascade Detection

Task: Predict if an event will trigger ANY cascade (binary classification)
Target: 1 if event triggers >= 1 cascade, 0 otherwise
"""

import numpy as np
import argparse
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_score,
    recall_score,
    roc_auc_score, 
    average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    """Configuration for binary cascade classifier."""
    
    # Data paths
    OUTPUT_DIR = Path('models/xgboost_binary')
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
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

def load_data(data_dir: Path):
    """Load prepared data and create binary targets."""
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_train_multilabel = np.load(data_dir / 'y_train.npy')
    y_test_multilabel = np.load(data_dir / 'y_test.npy')
    
    # Load metadata
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train (multilabel): {y_train_multilabel.shape}")
    print(f"  y_test (multilabel):  {y_test_multilabel.shape}")
    
    # Create binary targets: 1 if ANY cascade occurred, 0 otherwise
    y_train_binary = (y_train_multilabel.sum(axis=1) > 0).astype(int)
    y_test_binary = (y_test_multilabel.sum(axis=1) > 0).astype(int)
    
    print(f"\nBinary targets created:")
    print(f"  y_train_binary: {y_train_binary.shape}")
    print(f"  y_test_binary:  {y_test_binary.shape}")
    
    # Class distribution
    train_pos = (y_train_binary == 1).sum()
    train_neg = (y_train_binary == 0).sum()
    test_pos = (y_test_binary == 1).sum()
    test_neg = (y_test_binary == 0).sum()
    
    print(f"\nClass distribution:")
    print(f"  Train: {train_pos:,} positive ({train_pos/len(y_train_binary)*100:.2f}%), "
          f"{train_neg:,} negative ({train_neg/len(y_train_binary)*100:.2f}%)")
    print(f"  Test:  {test_pos:,} positive ({test_pos/len(y_test_binary)*100:.2f}%), "
          f"{test_neg:,} negative ({test_neg/len(y_test_binary)*100:.2f}%)")
    
    return (X_train, X_test, y_train_binary, y_test_binary, 
            feature_names, target_names)

def train_binary_classifier(X_train, y_train, config):
    """Train binary XGBoost classifier."""
    
    # Compute scale_pos_weight for class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1.0
    
    print(f"\nClass imbalance:")
    print(f"  Positive: {pos:,}")
    print(f"  Negative: {neg:,}")
    print(f"  Imbalance ratio: {neg/pos:.2f}:1")
    print(f"  scale_pos_weight: {scale:.2f}")
    
    # Create model
    params = config.XGBOOST_PARAMS.copy()
    if config.AUTO_SCALE_POS_WEIGHT:
        params['scale_pos_weight'] = scale
    
    model = XGBClassifier(**params)
    
    print(f"\nTraining XGBoost...")
    print(f"  Parameters: {params}")
    
    model.fit(X_train, y_train, verbose=False)
    
    return model

def predict(model, X_test):
    """Generate predictions."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return y_pred, y_prob

def evaluate(y_test, y_pred, y_prob):
    """Comprehensive binary classification evaluation."""
    
    results = {}
    
    # Basic metrics
    results['f1'] = f1_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred, zero_division=0)
    results['recall'] = recall_score(y_test, y_pred, zero_division=0)
    
    # AUC metrics
    results['roc_auc'] = roc_auc_score(y_test, y_prob)
    results['pr_auc'] = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results['true_negatives'] = int(tn)
    results['false_positives'] = int(fp)
    results['false_negatives'] = int(fn)
    results['true_positives'] = int(tp)
    
    # Additional metrics
    results['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, 
                                target_names=['No Cascade', 'Cascade'],
                                digits=4))
    
    print("\n--- Key Metrics ---")
    print(f"F1 Score:         {results['f1']:.4f}")
    print(f"Precision:        {results['precision']:.4f}")
    print(f"Recall:           {results['recall']:.4f}")
    print(f"Accuracy:         {results['accuracy']:.4f}")
    print(f"Specificity:      {results['specificity']:.4f}")
    
    print(f"\n--- AUC Metrics ---")
    print(f"ROC-AUC:          {results['roc_auc']:.4f}")
    print(f"PR-AUC:           {results['pr_auc']:.4f}")
    
    print(f"\n--- Confusion Matrix ---")
    print(f"True Negatives:   {tn:,}")
    print(f"False Positives:  {fp:,}")
    print(f"False Negatives:  {fn:,}")
    print(f"True Positives:   {tp:,}")
    
    return results, cm

def plot_results(cm, results, output_dir):
    """Create visualization plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Cascade', 'Cascade'],
                yticklabels=['No Cascade', 'Cascade'],
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Binary Cascade Detection')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc']
    metric_names = ['F1', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC', 'PR-AUC']
    values = [results[m] for m in metrics_to_plot]
    
    bars = ax.bar(metric_names, values, color=['#2ecc71', '#3498db', '#e74c3c', 
                                                '#f39c12', '#9b59b6', '#1abc9c'])
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Binary Cascade Detection - Performance Metrics')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_model(model, results, feature_names, output_dir):
    """Save model and results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'binary_cascade_model.json'
    model.save_model(model_path)
    print(f"\n✓ Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'xgboost_params': Config.XGBOOST_PARAMS,
            'auto_scale_pos_weight': Config.AUTO_SCALE_POS_WEIGHT
        },
        'results': results
    }
    
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save text report
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("Binary Cascade Detection Results\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        
        f.write("Metrics:\n")
        for key, value in results.items():
            if key not in ['true_negatives', 'false_positives', 
                          'false_negatives', 'true_positives']:
                f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(f"  True Negatives:  {results['true_negatives']:,}\n")
        f.write(f"  False Positives: {results['false_positives']:,}\n")
        f.write(f"  False Negatives: {results['false_negatives']:,}\n")
        f.write(f"  True Positives:  {results['true_positives']:,}\n")

def main(data_dir):
    """Run binary cascade detection pipeline."""
    
    config = Config()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names = \
        load_data(Path(data_dir))
    
    # Train
    model = train_binary_classifier(X_train, y_train, config)
    
    # Predict
    y_pred, y_prob = predict(model, X_test)
    
    # Evaluate
    results, cm = evaluate(y_test, y_pred, y_prob)
    
    # Visualize
    plot_results(cm, results, config.OUTPUT_DIR)
    
    # Save
    save_model(model, results, feature_names, config.OUTPUT_DIR)
    
    # Summary
    print(f"  F1 Score:     {results['f1']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    print(f"  ROC-AUC:      {results['roc_auc']:.4f}")
    print(f"  PR-AUC:       {results['pr_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Cascade Detection')
    parser.add_argument('--data_dir', type=str, help='Path to prepared data directory')
    args = parser.parse_args()
    main(args.data_dir)