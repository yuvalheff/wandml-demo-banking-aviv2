# Experiment 3: Cost-Sensitive Gradient Boosting

## Overview
**Objective**: Replace SMOTE oversampling with cost-sensitive learning using sample weights to address severe class imbalance and achieve >93% ROC-AUC and 60-70% recall targets.

**Key Change from Iteration 2**: Replace data augmentation (SMOTE) with algorithmic cost-sensitive approach using balanced sample weights.

## Rationale
Based on exploration experiments, cost-sensitive learning significantly outperforms SMOTE:
- **Cost-sensitive**: ROC-AUC 93.25%, Recall 86.01%, F1 60.11%
- **SMOTE (Iter 2)**: ROC-AUC 92.60%, Recall 52.17%, F1 56.73%

The approach addresses class imbalance at the algorithm level rather than data level, providing better performance and computational efficiency.

## Data Preprocessing

### 1. Target Variable Correction
```python
# Convert target from 1/2 encoding to 0/1 binary
train['target'] = train['target'] - 1
test['target'] = test['target'] - 1
```

### 2. Column Mapping
Rename feature columns from V1-V16 to original names:
```python
column_mapping = {
    'V1': 'age', 'V2': 'job', 'V3': 'marital', 'V4': 'education', 
    'V5': 'default', 'V6': 'balance', 'V7': 'housing', 'V8': 'loan',
    'V9': 'contact', 'V10': 'day', 'V11': 'month', 'V12': 'duration',
    'V13': 'campaign', 'V14': 'pdays', 'V15': 'previous', 'V16': 'poutcome'
}
```

### 3. Categorical Encoding
Apply LabelEncoder to categorical columns:
- `['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']`

### 4. Data Quality
- **Missing Values**: None present in dataset
- **Outliers**: No removal needed (tree-based models are robust)
- **Scaling**: Not required for Gradient Boosting

## Feature Engineering

Implement the same **44 engineered features** from Iteration 2 that proved effective:

### Duration-Based Features
- `log_duration = log(duration + 1)`
- `sqrt_duration = sqrt(duration)`
- `duration_binned` (short/medium/long/very_long based on quantiles)

### Balance-Related Features  
- `has_positive_balance = (balance > 0)`
- `balance_log_transform` (handle negative values)
- `balance_quartile` categorization

### Campaign Interaction Features
- `age_balance_ratio = age / (balance + 1000)` (handle negatives)
- `duration_per_contact = duration / campaign`
- `contact_efficiency` score

### Temporal Features
- Cyclical encodings: `day_sin, day_cos, month_sin, month_cos`
- Seasonal campaign patterns

### Demographic Combinations
- `job_education_combo`
- `marital_housing_combo` 
- Other relevant categorical interactions

### Previous Campaign Features
- `days_since_contact_binned` (pdays categorization)
- `previous_success_indicator`
- `campaign_history_score`

## Model Architecture

### Core Algorithm
**GradientBoostingClassifier** with cost-sensitive learning via sample weights

### Model Parameters
```python
GradientBoostingClassifier(
    n_estimators=200,           # Increased for better performance
    max_depth=8,                # Deeper trees for complex patterns
    learning_rate=0.05,         # Lower for stable learning
    subsample=0.8,              # Prevent overfitting
    max_features='sqrt',        # Feature randomness
    random_state=42,            # Reproducibility
    validation_fraction=0.1,    # Early stopping validation
    n_iter_no_change=20,       # Early stopping patience
    tol=1e-4                   # Convergence tolerance
)
```

### Cost-Sensitive Implementation
```python
# Calculate balanced sample weights
class_counts = np.bincount(y_train)
minority_weight = class_counts[0] / class_counts[1]  # ≈ 7.55

# Apply sample weights during training
sample_weights = np.where(y_train == 1, minority_weight, 1.0)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

### Validation Strategy
- **5-fold Stratified Cross-Validation** for robust performance estimates
- Same train/test split as previous iterations for fair comparison

## Evaluation Framework

### Primary Metrics
- **ROC-AUC** (target: >93.0%)
- **Recall** (target: 60-70%, stretch: 80%+)
- **Precision** (target: >45%)
- **F1-Score** (target: >55%)

### Secondary Metrics
- Balanced Accuracy
- Matthews Correlation Coefficient
- Specificity
- Average Precision (PR-AUC)

### Threshold Optimization
Evaluate performance at multiple thresholds:
- `[0.1, 0.2, 0.3, 0.4, 0.5]`
- Identify optimal threshold for business ROI
- Balance recall-precision trade-offs

### Error Analysis

#### 1. Confusion Matrix Analysis
- Detailed breakdown of TP, FP, TN, FN
- Focus on **reducing false negatives** (missed subscribers)
- Calculate business cost implications

#### 2. Feature Importance Analysis
- Rank top contributing features
- Compare feature rankings with Iteration 2
- Identify newly important features from cost-sensitive learning

#### 3. Prediction Confidence Analysis
- Examine probability distributions for both classes
- Identify low-confidence predictions
- Assess model certainty patterns

#### 4. Slice-Based Performance Analysis
- **Age Groups**: Young (<35), Middle (35-50), Senior (50+)
- **Job Categories**: Blue-collar, Management, Services, etc.
- **Previous Campaign Outcomes**: Success, Failure, Unknown
- **Balance Segments**: Negative, Low, Medium, High balance

#### 5. Calibration Assessment
- Generate calibration plots
- Calculate Brier score
- Assess probability reliability for business decisions

### Business Impact Metrics
- **Conversion Rate**: Percentage of targeted contacts resulting in subscriptions
- **Campaign Efficiency**: Contacts needed per successful subscription
- **Cost-Effectiveness**: Marketing cost per acquired customer
- **Coverage**: Percentage of potential subscribers identified

## Expected Outcomes

### Performance Targets
| Metric | Iteration 2 (SMOTE) | Iteration 3 Target | Stretch Goal |
|--------|-------------------|-------------------|--------------|
| ROC-AUC | 92.60% | **>93.0%** | 93.5% |
| Recall | 52.17% | **60-70%** | 80-86% |
| Precision | 62.16% | **>45%** | 45-50% |
| F1-Score | 56.73% | **>55%** | 58-62% |

### Business Impact
- **Subscriber Identification**: Detect 80-86% of potential subscribers (vs 52% in Iteration 2)
- **Campaign Efficiency**: Maintain reasonable precision while maximizing recall
- **Revenue Opportunity**: Reduce missed subscribers from 47.83% to ~15-20%

### Technical Benefits
- **Computational Efficiency**: Faster training without synthetic data generation
- **Memory Usage**: Lower memory requirements vs SMOTE
- **Model Interpretability**: Clear feature importance and decision boundaries
- **Deployment Ready**: Well-calibrated probabilities for threshold optimization

## Implementation Notes

### Key Differences from Iteration 2
1. **No Data Augmentation**: Replace SMOTE synthetic sampling with algorithmic weights
2. **Sample Weight Integration**: Apply weights during model training phase
3. **Same Feature Engineering**: Reuse proven 44-feature engineering pipeline
4. **Enhanced Evaluation**: More comprehensive threshold and business analysis

### Quality Assurance
- Use identical random seeds (`random_state=42`) for reproducibility
- Same train/test split for fair comparison with previous iterations
- Comprehensive logging of all hyperparameters and results

### Expected Timeline
- **Data Preprocessing**: 30 minutes
- **Feature Engineering**: 45 minutes  
- **Model Training**: 20 minutes
- **Evaluation & Analysis**: 60 minutes
- **Report Generation**: 30 minutes
- **Total**: ~3 hours

## Success Criteria
1. **ROC-AUC ≥ 93.0%** (exceed Iteration 2's 92.6%)
2. **Recall ≥ 60%** (substantially improve from 52.17%)
3. **Maintain F1-Score ≥ 55%** (balance precision-recall trade-off)
4. **Generate actionable business insights** for campaign optimization
5. **Deliver well-calibrated model** ready for production deployment