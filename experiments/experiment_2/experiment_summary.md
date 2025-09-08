# Experiment 2: SMOTE Enhanced Gradient Boosting - Results Summary

## Executive Summary

**Objective**: Improve recall from 46.1% to 60-70% while maintaining ROC-AUC > 0.93 using SMOTE oversampling technique.

**Key Results**: 
- **ROC-AUC**: 0.926 (slightly below 0.93 target)
- **Recall**: 52.17% (improved by 13.2% but below 60-70% target)
- **Precision**: 62.16% (exceeded 45% target)
- **F1-Score**: 56.73% (exceeded 55% target)

## Detailed Performance Metrics

### Primary Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ROC-AUC | 92.60% | >93% | ❌ (-0.40%) |
| Recall | 52.17% | 60-70% | ❌ (-7.83%) |
| Precision | 62.16% | >45% | ✅ (+17.16%) |
| F1-Score | 56.73% | >55% | ✅ (+1.73%) |

### Comprehensive Evaluation Results
- **Accuracy**: 90.69%
- **Balanced Accuracy**: 73.98%
- **Specificity**: 95.79%
- **Matthews Correlation Coefficient**: 0.5181
- **Average Precision (PR-AUC)**: 0.6144
- **Log Loss**: 0.2078
- **Brier Score**: 0.0642

### Confusion Matrix Analysis
```
                Predicted
              No      Yes    Total
Actual No   3,825    168    3,993  (95.79% specificity)
Actual Yes    253    276      529  (52.17% recall)
Total       4,078    444    4,522
```

**Business Impact**:
- **276 subscribers correctly identified** out of 529 potential (52.17%)
- **253 missed opportunities** (47.83% false negative rate)
- **62.16% precision** among flagged prospects
- **Class imbalance**: 7.55:1 ratio (challenging dataset)

## Iteration Comparison: Baseline vs SMOTE Enhancement

| Metric | Iteration 1 | Iteration 2 | Change |
|--------|------------|-------------|--------|
| ROC-AUC | 93.8% | 92.6% | -1.2% |
| Recall | 46.1% | 52.17% | +13.2% |
| Precision | 64.0% | 62.16% | -2.9% |
| F1-Score | 53.6% | 56.73% | +5.8% |

## Key Findings

### Achievements ✅
1. **Recall Improvement**: Successfully increased recall by 13.2% (46.1% → 52.17%)
2. **F1-Score Enhancement**: Achieved 5.8% improvement in balanced performance
3. **Target Compliance**: Met precision (>45%) and F1-score (>55%) targets
4. **High Specificity**: Maintained excellent 95.79% specificity
5. **Class Balance Handling**: SMOTE effectively addressed severe 11.7% minority class representation

### Challenges ❌
1. **ROC-AUC Decline**: Slight decrease from 93.8% to 92.6% (1.2% drop)
2. **Recall Gap**: Fell short of ambitious 60-70% recall target by 7.83%
3. **High False Negatives**: 47.83% of subscribers still missed
4. **Precision Trade-off**: Minor 2.9% precision decrease

### Technical Analysis
- **SMOTE Implementation**: Successfully balanced training data from 11.7% to ~50% minority class
- **Feature Engineering**: 44 engineered features from original 16 variables
- **Model Convergence**: Gradient boosting achieved stable out-of-bag score of 0.2059
- **Overfitting Assessment**: Balanced accuracy (73.98%) suggests good generalization

## Weaknesses and Limitations

1. **Recall Plateau**: SMOTE alone insufficient to achieve 60-70% recall target
2. **Synthetic Data Limitations**: SMOTE may not capture all minority class patterns
3. **ROC-AUC Trade-off**: Oversampling slightly reduced overall discriminative performance
4. **Class Imbalance Severity**: 7.55:1 ratio remains challenging for minority class detection

## Future Suggestions

### Priority 1: Advanced Sampling Strategies
- **ADASYN or BorderlineSMOTE**: More sophisticated synthetic data generation
- **Ensemble sampling**: Combine multiple oversampling techniques
- **Cost-sensitive learning**: Adjust class weights instead of data augmentation

### Priority 2: Threshold Optimization
- **Business-driven threshold**: Optimize for campaign ROI rather than balanced accuracy
- **Recall-precision trade-off analysis**: Find optimal operating point for business objectives
- **Multi-threshold evaluation**: Assess performance across different decision boundaries

### Priority 3: Advanced Feature Engineering
- **Temporal sequences**: Leverage previous campaign timing patterns
- **Interaction depth**: Explore 3-way feature interactions
- **Domain-specific features**: Industry knowledge-based feature creation

## Context Notes for Next Iteration

- **SMOTE Parameters**: Current k_neighbors=5 may benefit from tuning
- **Feature Importance**: Duration-related features maintained high importance (37.2%)
- **Data Quality**: No missing values, clean dataset for further experimentation
- **Model Stability**: Gradient boosting performed consistently across iterations
- **Business Context**: Portuguese banking campaign targeting term deposit subscriptions

## Artifacts Generated

- **Model Artifacts**: `trained_models.pkl`, `feature_processor.pkl`, `data_processor.pkl`
- **Visualizations**: ROC curve, precision-recall curve, confusion matrix, feature importance
- **MLflow Model**: Registered with signature and metadata for deployment
- **Evaluation Plots**: Calibration analysis and prediction distribution assessment