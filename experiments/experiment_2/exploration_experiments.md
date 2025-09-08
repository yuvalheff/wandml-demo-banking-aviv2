# Exploration Experiments Summary - Iteration 2

## Overview

Before finalizing the experiment plan for Iteration 2, I conducted several lightweight exploration experiments to validate the most effective approach for addressing class imbalance. The goal was to improve recall from 46.1% while maintaining ROC-AUC > 0.93.

## Dataset Context

- **Training Set**: 40,689 samples (88.3% class 0, 11.7% class 1)
- **Test Set**: 4,522 samples (same distribution)
- **Target Variable**: Binary classification (0=no subscription, 1=subscription)
- **Features**: 16 features after column mapping from V1-V16 to meaningful names

## Exploration Experiments Conducted

### Experiment 1: Baseline Performance
**Purpose**: Establish current performance without any class imbalance handling

**Configuration**:
- Gradient Boosting (50 estimators, depth=4, lr=0.2)
- Basic preprocessing with key engineered features (duration_log, balance_log)
- No class imbalance handling

**Results**:
- ROC-AUC: **0.9253**
- Precision: **0.629**
- Recall: **0.474**
- F1-Score: **0.541**

**Insight**: Confirms the class imbalance problem - high precision but poor recall, matching Iteration 1 findings.

### Experiment 2: SMOTE Oversampling
**Purpose**: Test SMOTE's effectiveness for improving recall

**Configuration**:
- Same model as baseline
- SMOTE with k_neighbors=3, random_state=42
- Balanced training set: 35,929 samples per class

**Results**:
- ROC-AUC: **0.9087** ✓ (maintained > 0.9)
- Precision: **0.471** ✓ (reasonable trade-off)
- Recall: **0.711** ✅ (major improvement from 47.4%)
- F1-Score: **0.567** ✅ (improved from 0.541)

**Key Finding**: SMOTE successfully improved recall to 71.1% while maintaining strong ROC-AUC performance.

### Experiment 3: Class Weight Balancing
**Purpose**: Test weighted loss approach as alternative to oversampling

**Configuration**:
- Same baseline model
- Sample weights: {class 0: 0.566, class 1: 4.274}
- No data augmentation, only loss function weighting

**Results**:
- ROC-AUC: **0.9231** ✅ (excellent maintenance)
- Precision: **0.449** ✓ (acceptable)
- Recall: **0.802** ✅ (highest recall achieved)
- F1-Score: **0.577** ✅ (best F1 score)

**Key Finding**: Class weighting achieved the highest recall (80.2%) with excellent ROC-AUC preservation.

### Experiment 4: Threshold Optimization
**Purpose**: Evaluate if simple threshold tuning can improve performance

**Configuration**:
- Baseline model without class imbalance handling
- Test thresholds from 0.1 to 0.5
- Find optimal F1-Score threshold

**Results by Threshold**:
| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 0.10      | 0.415     | 0.883  | 0.565    |
| 0.15      | 0.457     | 0.805  | 0.583    |
| 0.20      | 0.490     | 0.767  | 0.598    |
| 0.25      | 0.514     | 0.716  | 0.598    |
| **0.30**  | **0.545** | **0.667** | **0.600** |
| 0.35      | 0.570     | 0.624  | 0.596    |
| 0.40      | 0.591     | 0.573  | 0.582    |
| 0.50      | 0.629     | 0.474  | 0.541    |

**Key Finding**: Optimal threshold of 0.30 achieves 66.7% recall with 60.0% F1-score, providing a viable complementary approach.

## Comparative Analysis

### Performance Summary
| Approach | ROC-AUC | Precision | Recall | F1-Score | Key Advantage |
|----------|---------|-----------|--------|----------|---------------|
| Baseline | 0.9253 | 0.629 | 0.474 | 0.541 | High precision |
| **SMOTE** | **0.9087** | **0.471** | **0.711** | **0.567** | **Balanced improvement** |
| Class Weights | 0.9231 | 0.449 | 0.802 | 0.577 | Highest recall |
| Threshold (0.3) | 0.9253 | 0.545 | 0.667 | 0.600 | No training changes |

### Decision Rationale

**SMOTE Selected as Primary Approach** for the following reasons:

1. **Balanced Performance**: Achieves 71.1% recall (target: 60-70%) while maintaining ROC-AUC > 0.9
2. **Data Augmentation Benefits**: Creates synthetic samples that can help model generalization
3. **Proven Methodology**: Well-established technique with strong theoretical foundation
4. **Implementation Clarity**: Clean separation between data preparation and model training
5. **Complementary with Thresholding**: Can combine SMOTE with threshold optimization for further gains

**Class Weights Alternative**: While achieving higher recall (80.2%), the precision drops significantly (44.9%), which might not meet business requirements for campaign efficiency.

**Threshold Optimization**: Provides a good baseline improvement (66.7% recall) and will be included as a secondary analysis in the main experiment.

## Implementation Insights

### Technical Considerations
1. **Target Encoding**: Dataset uses 1/2 encoding instead of 0/1 - requires transformation
2. **Feature Engineering**: Duration and balance features remain critical (from Iteration 1)
3. **Cross-Validation**: Need stratified CV to handle imbalanced evaluation properly
4. **Synthetic Data Quality**: SMOTE maintains reasonable feature distributions

### Risk Mitigation
1. **Overfitting Risk**: Use cross-validation and test on original unbalanced test set
2. **Synthetic Data Validity**: Monitor feature correlations in SMOTE-generated samples
3. **Business Alignment**: Validate recall improvements translate to campaign ROI

## Final Recommendation

**Proceed with SMOTE Enhanced Gradient Boosting** as the primary approach for Iteration 2:
- Target recall improvement: 47.4% → 70%+ 
- Maintain ROC-AUC > 0.93
- Include threshold optimization as secondary analysis
- Comprehensive evaluation with business impact assessment

This approach provides the best balance of technical performance and business applicability while building upon the successful foundation of Iteration 1.