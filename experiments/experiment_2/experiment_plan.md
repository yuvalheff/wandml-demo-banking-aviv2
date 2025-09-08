# Experiment 2: SMOTE Enhanced Gradient Boosting

## Experiment Overview

**Objective**: Improve recall from 46.1% to 60-70% while maintaining ROC-AUC > 0.93 using SMOTE oversampling technique

**Primary Change**: Implementation of SMOTE (Synthetic Minority Oversampling Technique) to address severe class imbalance (88.3% non-subscribers vs 11.7% subscribers)

**Expected Impact**: Significantly improve minority class detection while preserving discriminative performance

## Background & Motivation

### Iteration 1 Results Summary
- ✅ **Excellent ROC-AUC**: 0.938 (exceeded target of >0.93)
- ✅ **Strong Feature Engineering**: Duration-based features dominated importance (37.2%)
- ❌ **Poor Recall**: Only 46.1% of actual subscribers identified
- ❌ **Class Imbalance Impact**: 96.6% specificity but missed 54% of subscribers

### Business Impact
The previous model's poor recall means missing potential subscribers, directly impacting campaign ROI. With only 46% subscriber identification, the bank loses significant revenue opportunities.

## Detailed Implementation Plan

### 1. Data Preprocessing

#### Target Variable Encoding
```python
# Convert from 1/2 encoding to 0/1 binary encoding
train_df['y'] = train_df['y'] - 1
test_df['y'] = test_df['y'] - 1
```

#### Categorical Feature Encoding

**High Cardinality Features** (One-hot encoding):
- `job` - 12 categories (blue-collar, management, technician, etc.)
- `education` - 4 categories (primary, secondary, tertiary, unknown)  
- `poutcome` - 4 categories (failure, other, success, unknown)

**Low Cardinality Features** (Label encoding):
- `marital` - 3 categories (divorced, married, single)
- `default`, `housing`, `loan` - Binary (no/yes)
- `contact` - 3 categories (cellular, telephone, unknown)
- `month` - 12 categories (jan through dec)

### 2. Feature Engineering

#### Duration Features (Top Priority)
```python
# Duration showed strongest correlation (0.395) and 37.2% model importance
df['duration_log'] = np.log1p(df['duration'])
df['duration_sqrt'] = np.sqrt(df['duration'])
```

#### Balance Features
```python
# Handle highly skewed distribution with 8.2% negative balances
df['balance_log'] = np.log1p(df['balance'] - df['balance'].min() + 1)
df['balance_positive'] = (df['balance'] > 0).astype(int)
```

#### Campaign & Temporal Features
```python
# Campaign efficiency metrics
df['campaign_per_previous'] = df['campaign'] / (df['previous'] + 1)
df['pdays_binary'] = (df['pdays'] != -1).astype(int)

# Cyclical temporal encoding
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
df['is_month_end'] = (df['day'] >= 28).astype(int)
```

#### Interaction Features
```python
# Meaningful demographic-behavioral interactions
df['age_balance_ratio'] = df['age'] / (df['balance'].abs() + 1)
df['duration_campaign_ratio'] = df['duration'] / (df['campaign'] + 1)
```

### 3. Class Imbalance Handling: SMOTE

#### Implementation Details
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy='auto',  # Balance to 1:1 ratio
    k_neighbors=5,            # Standard k-neighbors
    random_state=42           # Reproducibility
)

X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)
```

#### Expected Impact
- **Training Samples**: Minority class increases from 4,760 to ~35,929 samples
- **Recall Target**: Improve from 46.1% to 60-70%
- **ROC-AUC**: Maintain > 0.93 (exploration showed 0.909 achievable)

#### Validation Strategy
- Apply SMOTE only to training data
- Evaluate on original unbalanced test set for realistic performance assessment
- Compare synthetic sample quality with original minority class distribution

### 4. Model Configuration

#### Gradient Boosting Classifier
```python
GradientBoostingClassifier(
    n_estimators=200,      # Sufficient iterations for complex patterns
    max_depth=6,           # Prevent overfitting while capturing interactions
    learning_rate=0.1,     # Conservative learning for stability
    subsample=0.8,         # Stochastic sampling for regularization
    max_features='sqrt',   # Feature subsampling for generalization
    random_state=42        # Reproducibility
)
```

### 5. Evaluation Strategy

#### Cross-Validation
- **Method**: 5-fold Stratified Cross-Validation
- **Metrics**: ROC-AUC, Precision, Recall, F1-Score
- **Purpose**: Detect overfitting on SMOTE-enhanced training set

#### Test Set Evaluation

**Primary Metric**
- **ROC-AUC**: Target > 0.93 (maintain discriminative performance)

**Secondary Metrics**
- **Recall**: Target 60-70% (significant improvement from 46.1%)
- **Precision**: Target > 0.45 (reasonable false positive control)
- **F1-Score**: Target > 0.55 (balanced performance)

#### Diagnostic Analyses

1. **Confusion Matrix**: Analyze false positive/negative patterns
2. **ROC Curve**: Visualize discriminative performance across thresholds
3. **Precision-Recall Curve**: Assess performance trade-offs for imbalanced data
4. **Feature Importance**: Validate engineering success and SMOTE impact
5. **Calibration Plot**: Assess prediction confidence quality
6. **SMOTE Quality Analysis**: Compare synthetic vs original data distributions

#### Business Impact Analysis
- **Threshold Optimization**: Find optimal cutoff for campaign targeting
- **Cost-Benefit Analysis**: Calculate ROI using campaign costs vs conversion value
- **Coverage Analysis**: Assess percentage of potential subscribers identified

### 6. Expected Outputs

#### Model Artifacts
- Trained Gradient Boosting model with preprocessing pipeline
- SMOTE transformer for consistent data preparation
- Feature encoders for categorical variables
- MLflow model registration with version 2 tag

#### Evaluation Reports
- Performance comparison: Iteration 1 vs Iteration 2
- Feature importance analysis with SMOTE impact assessment  
- Business ROI analysis and campaign recommendations

#### Visualizations
- **Performance**: ROC curve, PR curve, confusion matrix, calibration plot
- **Features**: Feature importance, SMOTE distribution comparison
- **Business**: Threshold optimization, class balance impact analysis

### 7. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROC-AUC | ≥ 0.93 | Maintain discriminative performance |
| Recall | 60-70% | Significant improvement from 46.1% |
| Precision | ≥ 0.45 | Control false positive rate |
| F1-Score | ≥ 0.55 | Balanced performance metric |

### 8. Risk Mitigation

#### Overfitting Prevention
- 5-fold cross-validation for reliable performance estimation
- Regularization through subsample=0.8 and max_features='sqrt'
- Monitor validation metrics during training

#### Synthetic Data Quality
- Validate SMOTE synthetic samples don't create unrealistic combinations
- Ensure feature correlations are preserved in synthetic data
- Compare minority class patterns before/after SMOTE

### 9. Implementation Timeline

1. **Data Preprocessing** (30 min): Target encoding and categorical handling
2. **Feature Engineering** (45 min): Duration, balance, campaign, and interaction features
3. **SMOTE Application** (15 min): Generate balanced training set
4. **Model Training** (30 min): Fit Gradient Boosting with cross-validation
5. **Evaluation & Analysis** (60 min): Comprehensive performance assessment
6. **Visualization & Reporting** (45 min): Generate plots and business insights

**Total Estimated Time**: 3.5 hours

## Expected Business Impact

### Campaign Optimization
- **Improved Targeting**: 60-70% recall means identifying more potential subscribers
- **Better ROI**: Reduced missed opportunities while maintaining precision
- **Strategic Insights**: Enhanced feature importance for campaign design

### Model Deployment
- **Production Ready**: MLflow integration for seamless deployment
- **Monitoring**: Calibration plots enable confidence-based decision making
- **Scalability**: Pipeline structure supports future iterations and improvements

This experiment represents a focused, data-driven approach to solving the primary limitation of Iteration 1 while building upon its successful foundation.