#!/usr/bin/env python3
"""
Comprehensive script to extract evaluation metrics from trained gradient boosting model.
This script loads the trained model and processors, makes predictions on test data,
and calculates comprehensive evaluation metrics.
"""

import pickle
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
    average_precision_score, log_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

def load_artifacts(base_path):
    """Load all pickled artifacts."""
    artifacts = {}
    
    # Load trained model
    model_path = os.path.join(base_path, "trained_models.pkl")
    with open(model_path, 'rb') as f:
        artifacts['model'] = pickle.load(f)
    
    # Load data processor
    data_proc_path = os.path.join(base_path, "data_processor.pkl")
    with open(data_proc_path, 'rb') as f:
        artifacts['data_processor'] = pickle.load(f)
    
    # Load feature processor
    feature_proc_path = os.path.join(base_path, "feature_processor.pkl")
    with open(feature_proc_path, 'rb') as f:
        artifacts['feature_processor'] = pickle.load(f)
    
    return artifacts

def load_test_data(test_path):
    """Load and prepare test data."""
    try:
        test_data = pd.read_csv(test_path)
        print(f"Loaded test data with shape: {test_data.shape}")
        print(f"Test data columns: {list(test_data.columns)}")
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def apply_column_mapping(df):
    """Apply column mapping from V1-V16 to meaningful names."""
    column_mapping = {
        'V1': 'age',
        'V2': 'job', 
        'V3': 'marital',
        'V4': 'education',
        'V5': 'default',
        'V6': 'balance',
        'V7': 'housing',
        'V8': 'loan',
        'V9': 'contact',
        'V10': 'day',
        'V11': 'month',
        'V12': 'duration',
        'V13': 'campaign',
        'V14': 'pdays',
        'V15': 'previous',
        'V16': 'poutcome',
        'target': 'y'
    }
    
    return df.rename(columns=column_mapping)

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='binary'))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='binary'))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, average='binary'))
    
    # Precision and recall for both classes
    metrics['precision_class_0'] = float(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_class_1'] = float(precision_score(y_true, y_pred, pos_label=1))
    metrics['recall_class_0'] = float(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_class_1'] = float(recall_score(y_true, y_pred, pos_label=1))
    metrics['f1_score_class_0'] = float(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_score_class_1'] = float(f1_score(y_true, y_pred, pos_label=1))
    
    # ROC and PR metrics
    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    metrics['average_precision'] = float(average_precision_score(y_true, y_pred_proba))
    
    # Log loss
    try:
        # Create probability matrix for both classes
        y_pred_proba_both = np.column_stack([1 - y_pred_proba, y_pred_proba])
        metrics['log_loss'] = float(log_loss(y_true, y_pred_proba_both))
    except Exception as e:
        print(f"Warning: Could not calculate log_loss: {e}")
        metrics['log_loss'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }
    
    # Additional derived metrics
    tn, fp, fn, tp = cm.ravel()
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall_class_1']  # Same as recall for positive class
    metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # Balanced accuracy
    metrics['balanced_accuracy'] = float((metrics['sensitivity'] + metrics['specificity']) / 2)
    
    return metrics

def extract_model_info(model_wrapper):
    """Extract model configuration and training information."""
    model_info = {}
    
    # Model configuration
    if hasattr(model_wrapper, 'config'):
        config = model_wrapper.config
        model_info['model_type'] = getattr(config, 'model_type', 'unknown')
        model_info['model_params'] = getattr(config, 'model_params', {})
    
    # Sklearn model attributes
    if hasattr(model_wrapper, 'model'):
        sklearn_model = model_wrapper.model
        model_info['sklearn_params'] = {
            'n_estimators': getattr(sklearn_model, 'n_estimators', None),
            'max_depth': getattr(sklearn_model, 'max_depth', None),
            'learning_rate': getattr(sklearn_model, 'learning_rate', None),
            'random_state': getattr(sklearn_model, 'random_state', None),
            'n_features_in': getattr(sklearn_model, 'n_features_in_', None),
            'n_classes': getattr(sklearn_model, 'n_classes_', None)
        }
        
        # Feature names
        if hasattr(sklearn_model, 'feature_names_in_'):
            model_info['feature_names'] = sklearn_model.feature_names_in_.tolist()
        
        # Training scores if available
        if hasattr(sklearn_model, 'train_score_'):
            train_scores = sklearn_model.train_score_
            model_info['training_scores'] = {
                'initial_score': float(train_scores[0]) if len(train_scores) > 0 else None,
                'final_score': float(train_scores[-1]) if len(train_scores) > 0 else None,
                'score_improvement': float(train_scores[0] - train_scores[-1]) if len(train_scores) > 0 else None,
                'n_iterations': len(train_scores)
            }
        
        # Feature importance if available
        if hasattr(sklearn_model, 'feature_importances_'):
            importances = sklearn_model.feature_importances_
            feature_names = model_info.get('feature_names', [f'feature_{i}' for i in range(len(importances))])
            
            # Create feature importance ranking
            importance_data = list(zip(feature_names, importances))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            model_info['feature_importance'] = {
                'top_10_features': [
                    {'feature': name, 'importance': float(imp)} 
                    for name, imp in importance_data[:10]
                ],
                'all_features': [
                    {'feature': name, 'importance': float(imp)} 
                    for name, imp in importance_data
                ]
            }
    
    model_info['is_fitted'] = getattr(model_wrapper, 'is_fitted', False)
    
    return model_info

def calculate_class_distribution_metrics(y_true):
    """Calculate class distribution information."""
    unique, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    
    distribution = {}
    for cls, count in zip(unique, counts):
        distribution[f'class_{int(cls)}'] = {
            'count': int(count),
            'percentage': float(count / total * 100)
        }
    
    # Calculate imbalance ratio
    if len(counts) == 2:
        imbalance_ratio = float(max(counts) / min(counts))
        distribution['imbalance_ratio'] = imbalance_ratio
    
    return distribution

def main():
    # Paths
    base_path = "/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/experiments/experiment_1/output/model_artifacts/"
    test_data_path = "/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/data/test_set.csv"
    manifest_path = "/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/experiments/experiment_1/output/general_artifacts/manifest.json"
    output_path = os.path.join(base_path, "extracted_metrics.json")
    
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION METRICS EXTRACTION")
    print("="*70)
    
    # Load artifacts
    print("\\n1. Loading model artifacts...")
    try:
        artifacts = load_artifacts(base_path)
        model_wrapper = artifacts['model']
        data_processor = artifacts['data_processor']
        feature_processor = artifacts['feature_processor']
        print("✓ Successfully loaded all artifacts")
    except Exception as e:
        print(f"✗ Error loading artifacts: {e}")
        return
    
    # Load test data
    print("\\n2. Loading test data...")
    test_data = load_test_data(test_data_path)
    if test_data is None:
        print("✗ Could not load test data")
        return
    
    # Apply column mapping
    test_data = apply_column_mapping(test_data)
    
    # Separate features and target
    if 'y' in test_data.columns:
        X_test = test_data.drop('y', axis=1)
        y_test = test_data['y'].values
        
        # Convert target from (1,2) to (0,1) if needed
        if np.min(y_test) == 1 and np.max(y_test) == 2:
            y_test = y_test - 1
            print("✓ Converted target from (1,2) to (0,1)")
    else:
        print("✗ Target column 'y' not found in test data")
        return
    
    print(f"✓ Test set: {len(X_test)} samples, {len(X_test.columns)} features")
    print(f"✓ Class distribution: {np.bincount(y_test)}")
    
    # Process data through pipeline
    print("\\n3. Processing test data through pipeline...")
    try:
        # First apply data processor (handles encoding, etc.)
        X_processed = data_processor.transform(X_test)
        print(f"✓ Data processor applied, shape: {X_processed.shape}")
        
        # Then apply feature processor (creates engineered features)
        X_features = feature_processor.transform(X_processed)  
        print(f"✓ Feature processor applied, shape: {X_features.shape}")
        
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        print("Trying alternative approach...")
        
        # Try using the model wrapper's pipeline directly if available
        try:
            # Some model wrappers have a preprocessing pipeline
            if hasattr(model_wrapper, 'preprocess'):
                X_features = model_wrapper.preprocess(X_test)
                print(f"✓ Model wrapper preprocessing applied, shape: {X_features.shape}")
            else:
                print("✗ Could not find preprocessing method")
                return
        except Exception as e2:
            print(f"✗ Alternative preprocessing also failed: {e2}")
            return
    
    # Make predictions
    print("\\n4. Making predictions...")
    try:
        # Try using processed features first
        try:
            y_pred_proba = model_wrapper.predict_proba(X_features)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            print(f"✓ Generated predictions using processed features: {len(y_pred)} samples")
        except:
            # Fallback: try with raw data if model wrapper handles preprocessing
            y_pred_proba = model_wrapper.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            print(f"✓ Generated predictions using raw data: {len(y_pred)} samples")
            
        print(f"✓ Prediction distribution: {np.bincount(y_pred)}")
        print(f"✓ Probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        print(f"Model wrapper type: {type(model_wrapper)}")
        print(f"Model wrapper methods: {[m for m in dir(model_wrapper) if not m.startswith('_')]}")
        return
    
    # Calculate metrics
    print("\\n5. Calculating comprehensive metrics...")
    
    # Test set evaluation metrics
    test_metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
    
    # Model information
    model_info = extract_model_info(model_wrapper)
    
    # Class distribution
    class_distribution = calculate_class_distribution_metrics(y_test)
    
    # Load manifest metrics if available
    manifest_metrics = {}
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            if 'metric' in manifest:
                manifest_metrics['training_roc_auc'] = manifest['metric']['value']
                manifest_metrics['metric_name'] = manifest['metric']['name']
    except Exception as e:
        print(f"Warning: Could not load manifest metrics: {e}")
    
    # Compile all metrics
    comprehensive_metrics = {
        'experiment_info': {
            'experiment_name': 'Gradient Boosting with Feature Engineering',
            'model_type': 'GradientBoostingClassifier',
            'task_type': 'binary_classification',
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'model_configuration': model_info,
        'test_set_evaluation': test_metrics,
        'training_metrics': manifest_metrics,
        'data_info': {
            'test_set_size': len(y_test),
            'n_features': len(X_test.columns) if X_test is not None else None,
            'class_distribution': class_distribution
        },
        'performance_summary': {
            'primary_metric_roc_auc': test_metrics['roc_auc'],
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'balanced_accuracy': test_metrics['balanced_accuracy']
        }
    }
    
    # Display summary
    print("\\n6. EVALUATION RESULTS SUMMARY:")
    print("-" * 50)
    print(f"ROC-AUC Score: {test_metrics['roc_auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print()
    print("Confusion Matrix:")
    cm = test_metrics['confusion_matrix']
    print(f"  True Negatives: {cm['true_negatives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  False Negatives: {cm['false_negatives']}")  
    print(f"  True Positives: {cm['true_positives']}")
    
    if 'feature_importance' in model_info:
        print("\\nTop 10 Most Important Features:")
        for i, feat in enumerate(model_info['feature_importance']['top_10_features'], 1):
            print(f"  {i:2d}. {feat['feature']}: {feat['importance']:.4f}")
    
    # Save comprehensive metrics
    print(f"\\n7. Saving comprehensive metrics to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        print("✓ Successfully saved comprehensive metrics")
    except Exception as e:
        print(f"✗ Error saving metrics: {e}")
        return
    
    print("\\n" + "="*70)
    print("METRICS EXTRACTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"All metrics saved to: {output_path}")

if __name__ == "__main__":
    main()