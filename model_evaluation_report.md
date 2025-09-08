# Gradient Boosting Model - Comprehensive Evaluation Report

**Experiment:** Gradient Boosting with Feature Engineering  
**Model Type:** GradientBoostingClassifier  
**Task:** Binary Classification (Bank Marketing - Term Deposit Prediction)  
**Date:** 2025-09-08

---

## Model Configuration

**Hyperparameters:**
- Number of estimators: 200
- Maximum depth: 6  
- Learning rate: 0.1
- Random state: 42
- Features used: 48 (after feature engineering)

**Training Performance:**
- Training iterations: 200
- Initial training score: 0.6578
- Final training score: 0.2612
- Score improvement: 0.3966

---

## Test Set Performance Metrics

### Primary Metrics
| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.9379** |
| Accuracy | 0.9067 |
| Balanced Accuracy | 0.7135 |
| Precision | 0.6404 |
| Recall (Sensitivity) | 0.4612 |
| F1-Score | 0.5363 |
| Specificity | 0.9657 |

### Class-Specific Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|---------|----------|
| Class 0 (No deposit) | 0.9312 | 0.9657 | 0.9481 |
| Class 1 (Deposit) | 0.6404 | 0.4612 | 0.5363 |

### Additional Metrics
- **Average Precision:** 0.6312
- **Log Loss:** 0.1964
- **False Positive Rate:** 0.0343
- **False Negative Rate:** 0.5388

---

## Confusion Matrix

|                | Predicted No | Predicted Yes | Total |
|----------------|--------------|---------------|-------|
| **Actual No**  | 3856         | 137           | 3993  |
| **Actual Yes** | 285          | 244           | 529   |
| **Total**      | 4141         | 381           | 4522  |

**Breakdown:**
- True Negatives: 3,856
- False Positives: 137  
- False Negatives: 285
- True Positives: 244

---

## Dataset Information

**Test Set:**
- Total samples: 4,522
- Original features: 16
- Engineered features: 48

**Class Distribution:**
- Class 0 (No deposit): 3,993 samples (88.30%)
- Class 1 (Deposit): 529 samples (11.70%)
- **Imbalance ratio:** 7.55:1

---

## Feature Importance Analysis

### Top 10 Most Important Features

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | duration_log | 0.2200 | Engineered |
| 2 | duration | 0.1518 | Original |
| 3 | pdays | 0.0829 | Original |
| 4 | poutcome | 0.0825 | Original |
| 5 | age | 0.0637 | Original |
| 6 | day | 0.0625 | Original |
| 7 | contact | 0.0363 | Original |
| 8 | housing | 0.0324 | Original |
| 9 | balance | 0.0314 | Original |
| 10 | month_mar | 0.0242 | Encoded |

### Key Insights:
- **Duration-related features** are the most predictive (combined importance: ~37%)
- **Feature engineering was effective:** `duration_log` is the single most important feature
- **Temporal factors matter:** Previous contact patterns (`pdays`, `poutcome`) are highly influential
- **Customer characteristics:** Age, housing situation, and contact preferences are significant predictors

---

## Model Performance Analysis

### Strengths:
1. **Excellent ROC-AUC (0.9379):** Model has strong discriminative ability
2. **High Specificity (0.9657):** Very good at identifying customers who won't subscribe
3. **Good Overall Accuracy (0.9067):** Correctly classifies ~91% of cases
4. **Low False Positive Rate (0.0343):** Minimal wasted marketing effort

### Areas for Improvement:
1. **Moderate Recall (0.4612):** Missing ~54% of potential subscribers
2. **Class Imbalance Impact:** Performance skewed toward majority class
3. **Precision-Recall Trade-off:** Could optimize threshold for better recall

### Business Impact:
- **Marketing Efficiency:** High precision (64%) means targeted campaigns will be reasonably effective
- **Opportunity Cost:** High false negative rate means missing potential customers
- **ROI Considerations:** Model excels at avoiding non-interested customers but could capture more interested ones

---

## Recommendations

### Model Optimization:
1. **Threshold Tuning:** Lower classification threshold to improve recall
2. **Cost-Sensitive Learning:** Adjust for business cost of false negatives vs. false positives
3. **Ensemble Methods:** Combine with other algorithms to improve minority class detection

### Feature Engineering:
1. **Duration features proved most valuable** - consider additional time-based features
2. **Previous campaign outcome** is highly predictive - enrich this information
3. **Seasonal patterns** show some importance - explore more temporal features

### Business Application:
1. **Use probability scores** rather than binary predictions for campaign prioritization
2. **Segment customers** based on probability ranges for different marketing strategies
3. **A/B test** different probability thresholds to optimize business outcomes

---

## Files Generated

1. **`extracted_metrics.json`** - Complete metrics in JSON format
2. **`extract_comprehensive_metrics.py`** - Python script for metrics extraction
3. **`model_evaluation_report.md`** - This comprehensive report

---

*Report generated automatically from trained model evaluation on 2025-09-08*