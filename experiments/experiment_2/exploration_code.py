#!/usr/bin/env python3
"""
Exploration experiments for class imbalance handling approaches
Iteration 2: Focus on improving recall from 46.1% while maintaining ROC-AUC > 0.93
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
train_path = '/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/data/train.csv'
test_path = '/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/data/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Dataset shape:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")

# Map columns based on EDA context
column_mapping = {
    'V1': 'age', 'V2': 'job', 'V3': 'marital', 'V4': 'education', 'V5': 'default',
    'V6': 'balance', 'V7': 'housing', 'V8': 'loan', 'V9': 'contact', 'V10': 'day',
    'V11': 'month', 'V12': 'duration', 'V13': 'campaign', 'V14': 'pdays',
    'V15': 'previous', 'V16': 'poutcome', 'target': 'y'
}

train_df.rename(columns=column_mapping, inplace=True)
test_df.rename(columns=column_mapping, inplace=True)

print("\nTarget distribution in train:")
print(train_df['y'].value_counts(normalize=True))

# Basic preprocessing function (from previous iteration)
def preprocess_data(df):
    # Create copy
    df_proc = df.copy()
    
    # Handle categorical encoding
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    for col in categorical_cols:
        if col in ['job', 'education', 'poutcome']:  # High cardinality - one-hot
            dummies = pd.get_dummies(df_proc[col], prefix=col, drop_first=False)
            df_proc = pd.concat([df_proc, dummies], axis=1)
            df_proc.drop(col, axis=1, inplace=True)
        else:  # Low cardinality - label encoding
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    
    # Feature engineering (key successful features from iteration 1)
    df_proc['duration_log'] = np.log1p(df_proc['duration'])
    df_proc['duration_sqrt'] = np.sqrt(df_proc['duration'])
    df_proc['balance_log'] = np.log1p(df_proc['balance'] - df_proc['balance'].min() + 1)
    df_proc['age_balance_ratio'] = df_proc['age'] / (df_proc['balance'].abs() + 1)
    df_proc['campaign_per_previous'] = df_proc['campaign'] / (df_proc['previous'] + 1)
    
    # Seasonal features
    df_proc['month_sin'] = np.sin(2 * np.pi * df_proc['day'] / 31)
    df_proc['month_cos'] = np.cos(2 * np.pi * df_proc['day'] / 31)
    
    return df_proc

# Preprocess train and test
X_train_proc = preprocess_data(train_df.drop('y', axis=1))
y_train = train_df['y']
X_test_proc = preprocess_data(test_df.drop('y', axis=1))
y_test = test_df['y']

print(f"Processed features: {X_train_proc.shape[1]}")

# Base model for comparison (from iteration 1)
base_model = GradientBoostingClassifier(
    n_estimators=100,  # Reduced for faster exploration
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("\n=== EXPLORATION EXPERIMENT 1: Baseline (No Sampling) ===")
cv_scores = cross_val_score(base_model, X_train_proc, y_train, cv=3, scoring='roc_auc')
print(f"Baseline ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

base_model.fit(X_train_proc, y_train)
y_pred = base_model.predict(X_test_proc)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Test Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print("\n=== EXPLORATION EXPERIMENT 2: SMOTE Oversampling ===")
smote = SMOTE(random_state=42, k_neighbors=3)
X_smote, y_smote = smote.fit_resample(X_train_proc, y_train)
print(f"SMOTE - Original: {Counter(y_train)}, Resampled: {Counter(y_smote)}")

cv_scores = cross_val_score(base_model, X_smote, y_smote, cv=3, scoring='roc_auc')
print(f"SMOTE ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

base_model.fit(X_smote, y_smote)
y_pred = base_model.predict(X_test_proc)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Test Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print("\n=== EXPLORATION EXPERIMENT 3: ADASYN Oversampling ===")
try:
    adasyn = ADASYN(random_state=42, n_neighbors=3)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train_proc, y_train)
    print(f"ADASYN - Original: {Counter(y_train)}, Resampled: {Counter(y_adasyn)}")
    
    cv_scores = cross_val_score(base_model, X_adasyn, y_adasyn, cv=3, scoring='roc_auc')
    print(f"ADASYN ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    base_model.fit(X_adasyn, y_adasyn)
    y_pred = base_model.predict(X_test_proc)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Test Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
except Exception as e:
    print(f"ADASYN failed: {e}")

print("\n=== EXPLORATION EXPERIMENT 4: BorderlineSMOTE ===")
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
X_borderline, y_borderline = borderline_smote.fit_resample(X_train_proc, y_train)
print(f"BorderlineSMOTE - Original: {Counter(y_train)}, Resampled: {Counter(y_borderline)}")

cv_scores = cross_val_score(base_model, X_borderline, y_borderline, cv=3, scoring='roc_auc')
print(f"BorderlineSMOTE ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

base_model.fit(X_borderline, y_borderline)
y_pred = base_model.predict(X_test_proc)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Test Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print("\n=== EXPLORATION EXPERIMENT 5: SMOTE + Tomek Links ===")
smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=3))
X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train_proc, y_train)
print(f"SMOTE+Tomek - Original: {Counter(y_train)}, Resampled: {Counter(y_smote_tomek)}")

cv_scores = cross_val_score(base_model, X_smote_tomek, y_smote_tomek, cv=3, scoring='roc_auc')
print(f"SMOTE+Tomek ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

base_model.fit(X_smote_tomek, y_smote_tomek)
y_pred = base_model.predict(X_test_proc)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Test Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print("\n=== EXPLORATION EXPERIMENT 6: Class Weight Balancing ===")
balanced_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    # Calculate class weights
    init='zero'  # Using balanced approach via sample weights during fit
)

# Calculate class weights
class_weights = {0: 1, 1: len(y_train[y_train==0]) / len(y_train[y_train==1])}
sample_weights = np.array([class_weights[y] for y in y_train])

print(f"Class weights: {class_weights}")

# Fit with sample weights
balanced_model.fit(X_train_proc, y_train, sample_weight=sample_weights)
y_pred = balanced_model.predict(X_test_proc)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Class-weighted Test Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# Test ROC-AUC
y_pred_proba = balanced_model.predict_proba(X_test_proc)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Class-weighted ROC-AUC: {roc_auc:.4f}")

print("\n=== EXPLORATION EXPERIMENT 7: Threshold Optimization ===")
base_model.fit(X_train_proc, y_train)
y_pred_proba = base_model.predict_proba(X_test_proc)[:, 1]

# Test different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
print("Threshold optimization results:")
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh, average='binary')
    print(f"Threshold {thresh}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print("\n=== EXPLORATION COMPLETE ===")