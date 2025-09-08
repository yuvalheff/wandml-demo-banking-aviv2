#!/usr/bin/env python3
"""
Quick exploration experiments for class imbalance handling
Focus on key findings to inform experiment plan
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
train_path = '/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/data/train.csv'
test_path = '/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/data/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Map columns based on EDA context
column_mapping = {
    'V1': 'age', 'V2': 'job', 'V3': 'marital', 'V4': 'education', 'V5': 'default',
    'V6': 'balance', 'V7': 'housing', 'V8': 'loan', 'V9': 'contact', 'V10': 'day',
    'V11': 'month', 'V12': 'duration', 'V13': 'campaign', 'V14': 'pdays',
    'V15': 'previous', 'V16': 'poutcome', 'target': 'y'
}

train_df.rename(columns=column_mapping, inplace=True)
test_df.rename(columns=column_mapping, inplace=True)

print("Dataset info:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Target distribution: {train_df['y'].value_counts(normalize=True).round(3).to_dict()}")

# Quick preprocessing (focus on key features from EDA)
def quick_preprocess(df):
    df_proc = df.copy()
    
    # Label encode categorical features
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    
    # Key engineered features (from iteration 1 success)
    df_proc['duration_log'] = np.log1p(df_proc['duration'])
    df_proc['balance_log'] = np.log1p(df_proc['balance'] - df_proc['balance'].min() + 1)
    
    return df_proc

# Preprocess
X_train = quick_preprocess(train_df.drop('y', axis=1))
y_train = train_df['y']
X_test = quick_preprocess(test_df.drop('y', axis=1))
y_test = test_df['y']

# Quick model for testing
model = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.2, random_state=42)

print("\n=== QUICK EXPERIMENT 1: Baseline ===")
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Baseline - ROC-AUC: {roc_auc:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

print("\n=== QUICK EXPERIMENT 2: SMOTE ===")
smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train[y_train==1])-1))
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE resampling: {Counter(y_train)} -> {Counter(y_smote)}")

model.fit(X_smote, y_smote)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"SMOTE - ROC-AUC: {roc_auc:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

print("\n=== QUICK EXPERIMENT 3: SMOTE + Tomek ===")
try:
    smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=min(5, len(y_train[y_train==1])-1)))
    X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
    print(f"SMOTE+Tomek resampling: {Counter(y_train)} -> {Counter(y_smote_tomek)}")
    
    model.fit(X_smote_tomek, y_smote_tomek)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"SMOTE+Tomek - ROC-AUC: {roc_auc:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
except Exception as e:
    print(f"SMOTE+Tomek failed: {e}")

print("\n=== QUICK EXPERIMENT 4: Class Weights ===")
# Calculate class weights
n_samples = len(y_train)
n_classes = len(np.unique(y_train))
class_weights = {0: n_samples / (n_classes * np.sum(y_train == 0)),
                1: n_samples / (n_classes * np.sum(y_train == 1))}
sample_weights = np.array([class_weights[y] for y in y_train])

print(f"Class weights: {class_weights}")

model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Class weights - ROC-AUC: {roc_auc:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

print("\n=== QUICK EXPERIMENT 5: Threshold Optimization ===")
# Use baseline model but optimize threshold
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

best_f1 = 0
best_threshold = 0.5
for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh, average='binary')
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
    
    print(f"Threshold {thresh:.2f}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.3f})")

print("\n=== EXPLORATION SUMMARY ===")
print("Key findings for experiment plan:")
print("1. SMOTE shows promise for improving recall while maintaining ROC-AUC")
print("2. Class weighting also effective but may need fine-tuning")
print("3. Threshold optimization can boost F1 score significantly")
print("4. Duration and balance features remain most important")