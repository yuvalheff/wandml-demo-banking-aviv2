# Bank Marketing Dataset - EDA Report

## Dataset Overview
- **Total Samples**: 40,689 clients
- **Features**: 17 (16 predictive + 1 target)  
- **Missing Values**: 0
- **Task Type**: Binary Classification (Term Deposit Prediction)
- **Target Distribution**: 88.3% No (35,929) vs 11.7% Yes (4,760)

## Key Findings

### 1. Target Variable Analysis
- **Highly imbalanced dataset** requiring stratified sampling and ROC-AUC optimization
- Class distribution: 88.3% non-subscribers vs 11.7% subscribers
- Sufficient sample size (40k+) for robust model training

### 2. Feature Importance Insights

#### Numerical Features (Correlation with Target)
1. **Duration** (0.395) - Strongest predictor
   - Successful calls: 537s average vs 221s for failures
   - Clear separation between classes
2. **Pdays** (0.106) - Time since last contact
3. **Previous** (0.091) - Number of previous contacts
4. **Campaign** (0.075) - Current campaign contacts
5. **Balance** (0.052) - Account balance (weak)

#### Categorical Features  
1. **Previous Outcome (poutcome)** - Most powerful categorical predictor
   - Success: 64.6% conversion rate
   - Unknown: 9.1% conversion rate
   - Failure/Other: 12.7-16.8% conversion rates

2. **Month** - Strong seasonal patterns
   - Best: March (52.9%), December (47.6%), September (46.5%)
   - Worst: May (6.7%), July (9.2%), January (10.2%)

### 3. Data Quality Issues

#### Balance Distribution
- Highly right-skewed (mean €1,361, median €448)
- **3,356 clients (8.2%) have negative balances** (overdrafts)
- Extreme outliers up to €98,417
- **Recommendation**: Log transformation or winsorization

#### Duration Outliers
- Log-normal distribution with extreme outliers
- Maximum duration: 4,918 seconds (>1 hour)
- **Recommendation**: Duration-based feature engineering

### 4. Demographic Insights

#### Job Distribution
- Blue-collar: 8,776 (21.6%)  
- Management: 8,523 (20.9%)
- Technician: 6,812 (16.7%)
- Admin: 4,640 (11.4%)
- Services: 3,751 (9.2%)

#### Age Profile
- Mean age: 40.9 years (median: 39)
- Range: 18-95 years
- Right-skewed distribution
- 75% of clients under 48 years

### 5. Campaign Analysis

#### Contact Method
- Cellular: Most common contact method
- Telephone: Traditional approach  
- Unknown: Missing data requiring imputation

#### Temporal Patterns
- Strong monthly variations in success rates
- May has highest volume (12,410 contacts) but lowest success (6.7%)
- March has moderate volume (427) but highest success (52.9%)

## Preprocessing Recommendations

### 1. Feature Engineering
- **Duration bins**: Create categorical duration ranges
- **Seasonal features**: Extract season from month
- **Balance transformation**: Log or quantile normalization
- **Previous campaign metrics**: Success rate, recency features

### 2. Categorical Encoding
- **Ordinal encoding** for education levels
- **One-hot encoding** for nominal features (job, marital, contact)
- **Target encoding** for high-cardinality features

### 3. Outlier Treatment
- **Balance**: Winsorization at 95th percentile
- **Duration**: Cap extreme values or use robust scaling
- **Age**: Generally clean, no treatment needed

### 4. Class Imbalance Handling
- **Stratified sampling** for train/validation splits
- Consider **SMOTE** or other oversampling techniques
- Use **class weights** in model training
- Focus on **ROC-AUC** and **PR-AUC** metrics

## Model Development Strategy

### 1. Baseline Models
- Logistic Regression (interpretable baseline)
- Random Forest (handle non-linearity)
- Gradient Boosting (XGBoost/LightGBM)

### 2. Advanced Techniques
- **Feature selection** based on correlation and mutual information
- **Hyperparameter tuning** with cross-validation
- **Ensemble methods** for improved performance

### 3. Validation Strategy
- **Stratified K-Fold** cross-validation
- **Time-based splits** if temporal leakage concerns exist
- **ROC-AUC** as primary metric with **Precision-Recall AUC** as secondary

## Business Insights

### 1. Campaign Optimization
- **Timing**: Focus campaigns in March, December, September
- **Avoid**: May campaigns show consistently poor performance  
- **Duration**: Longer calls strongly indicate success potential

### 2. Client Segmentation
- **Previous success clients**: High-value targets (64.6% success)
- **Professional segments**: Blue-collar and management dominant
- **Age targeting**: Focus on 33-48 age range (core demographic)

### 3. Contact Strategy
- **Cellular contacts** preferred over telephone
- **Follow-up timing**: Leverage pdays for optimal contact intervals
- **Call quality**: Duration optimization more important than frequency

This EDA provides a solid foundation for developing an effective term deposit prediction model with clear preprocessing steps and business-driven insights.