# Experiment 1: Gradient Boosting with Feature Engineering

## Experiment Overview
**Experiment Name:** Gradient Boosting with Feature Engineering  
**Task Type:** Binary Classification  
**Target Column:** y (term deposit subscription: 0=no, 1=yes)  
**Primary Metric:** ROC-AUC  

## Data Preprocessing Steps

### 1. Data Loading & Column Mapping
- Load train/test datasets from CSV files
- Map generic column names to actual features:
  - V1=age, V2=job, V3=marital, V4=education, V5=default
  - V6=balance, V7=housing, V8=loan, V9=contact, V10=day
  - V11=month, V12=duration, V13=campaign, V14=pdays
  - V15=previous, V16=poutcome, target=y
- Convert target variable from (1,2) to (0,1) by subtracting 1

### 2. Categorical Encoding Strategy
- **High-cardinality features** ('job', 'month'): One-hot encoding with drop_first=True
- **Low-cardinality features** ('marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'): Label encoding
- No scaling required for tree-based models
- No outlier clipping (experiments showed minimal benefit)

## Feature Engineering Steps

### 1. Duration-Based Features (Strongest Predictor)
- `duration_log` = log1p(duration) - handles skewness
- `duration_short` = (duration < 100) - very short calls indicator
- `duration_long` = (duration > 500) - successful call length indicator

### 2. Balance-Based Features (Handle Skewness & Negatives)
- `balance_log` = log(balance) where balance > 0, else 0
- `balance_negative` = (balance < 0) - overdraft indicator
- `balance_zero` = (balance == 0) - zero balance indicator

### 3. Campaign Features
- `campaign_multiple` = (campaign > 1) - multiple contact indicator
- `pdays_contacted_before` = (pdays != -1) - previous contact indicator

### 4. Seasonal Features (Based on EDA Insights)
- `month_high_success` = month in ['mar', 'dec', 'sep'] - high success months
- `month_low_success` = month in ['may', 'jul', 'jun'] - low success months

### 5. Previous Outcome Features (Most Powerful Categorical Predictor)
- `poutcome_success` = (poutcome == 'success') - previous success indicator
- `poutcome_unknown` = (poutcome == 'unknown') - unknown outcome indicator

**Final Dataset Shape:** 48 features after preprocessing and encoding

## Model Selection & Training

### Primary Model: Gradient Boosting Classifier
**Selection Rationale:** Achieved highest ROC-AUC of 0.9335 in cross-validation experiments

**Hyperparameters:**
- n_estimators=200
- max_depth=6  
- learning_rate=0.1
- random_state=42

### Training Strategy
1. 5-fold stratified cross-validation for model evaluation
2. Stratification maintains class balance (88.3% vs 11.7%) across folds
3. Train final model on complete training set
4. Evaluate on holdout test set

### Baseline Comparisons
- Random Forest (ROC-AUC: 0.9250)
- Extra Trees (ROC-AUC: 0.9222)

## Evaluation Strategy

### Primary Evaluation
- **Main Metric:** ROC-AUC (optimal for imbalanced binary classification)
- **Cross-Validation:** 5-fold stratified CV
- **Test Set Evaluation:** Final performance on holdout data

### Comprehensive Analysis Framework

#### 1. Model Performance Metrics
- ROC-AUC, Precision, Recall, F1-Score (for both classes)
- Confusion matrix analysis
- ROC curve and Precision-Recall curve visualization

#### 2. Feature Analysis
- Feature importance ranking
- Top 10 most predictive features identification
- Feature contribution analysis

#### 3. Model Diagnostics
- Probability calibration analysis and calibration plots
- Learning curves to assess overfitting/underfitting
- Validation curves for key hyperparameters

#### 4. Segmented Performance Analysis
- **By Demographics:** Performance across age groups and job types
- **By Campaign Characteristics:** Analysis by month, contact type, campaign frequency
- **By Financial Profile:** Performance across balance quartiles

#### 5. Error Analysis
- Misclassification pattern identification
- Analysis of prediction confidence distribution
- Duration threshold impact on predictions
- Seasonal performance variation analysis

#### 6. Business Insights
- Identification of high-value customer segments
- Campaign timing optimization recommendations
- Feature-based targeting strategies

## Expected Outputs

### Model Artifacts
1. **Trained model:** GradientBoostingClassifier saved as .pkl file
2. **Feature encoders:** Label encoders and column transformers saved for inference
3. **Preprocessing pipeline:** Complete feature engineering pipeline

### Performance Reports  
1. **Model evaluation report:** Comprehensive metrics and diagnostics
2. **Feature importance analysis:** Ranked feature contributions
3. **Segmented performance analysis:** Performance by key business dimensions
4. **ROC/PR curves:** Model discrimination visualizations
5. **Calibration analysis:** Probability reliability assessment

### Predictions & Business Intelligence
1. **Test set predictions:** CSV with probabilities and class predictions  
2. **Business insights report:** Actionable recommendations for campaign optimization
3. **Model performance dashboard:** Key metrics and visualizations

### Expected Performance
- **Target ROC-AUC:** >0.93 (based on cross-validation results)
- **Baseline Improvement:** ~0.7% over Random Forest baseline
- **Class Balance Handling:** Effective prediction despite 88.3%/11.7% imbalance

## Implementation Notes
- Dataset is already split (train: 40,689 samples, test: 4,522 samples)
- No missing values present in dataset
- All preprocessing maintains data integrity and business interpretability
- Model selection based on rigorous cross-validation comparison
- Evaluation framework designed for comprehensive model understanding and business value extraction