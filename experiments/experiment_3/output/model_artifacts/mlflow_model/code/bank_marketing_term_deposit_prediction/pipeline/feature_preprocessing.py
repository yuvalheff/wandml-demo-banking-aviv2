from typing import Optional
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

from bank_marketing_term_deposit_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.is_fitted = False
        self.balance_min = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # Store balance minimum for log transformation
        if 'balance' in X.columns:
            self.balance_min = X['balance'].min()
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.
        
        Implements all 44 engineered features from iteration 2 experiment plan.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
            
        X_transformed = X.copy()
        
        # 1. Duration transformations (3 features)
        if self.config.enable_duration_features and 'duration' in X_transformed.columns:
            X_transformed['log_duration'] = np.log1p(X_transformed['duration'])
            X_transformed['sqrt_duration'] = np.sqrt(X_transformed['duration'])
            # Duration binned - quantile-based categories
            X_transformed['duration_binned'] = pd.qcut(X_transformed['duration'], q=5, labels=False, duplicates='drop')
        
        # 2. Balance indicators (3 features)
        if self.config.enable_balance_features and 'balance' in X_transformed.columns:
            X_transformed['has_positive_balance'] = (X_transformed['balance'] > 0).astype(int)
            # Handle negative balances for log transformation
            X_transformed['balance_log_transform'] = np.log1p(X_transformed['balance'] - self.balance_min + 1)
            X_transformed['balance_quartile'] = pd.qcut(X_transformed['balance'], q=4, labels=False, duplicates='drop')
        
        # 3. Campaign interactions (3 features)
        if self.config.enable_campaign_features:
            if 'age' in X_transformed.columns and 'balance' in X_transformed.columns:
                X_transformed['age_balance_ratio'] = X_transformed['age'] / (np.abs(X_transformed['balance']) + 1)
            if 'duration' in X_transformed.columns and 'campaign' in X_transformed.columns:
                X_transformed['duration_per_contact'] = X_transformed['duration'] / (X_transformed['campaign'] + 1)
            if 'duration' in X_transformed.columns and 'campaign' in X_transformed.columns:
                X_transformed['contact_efficiency'] = X_transformed['duration'] * X_transformed['campaign']
        
        # 4. Temporal features - cyclical encodings (6 features)
        if self.config.enable_seasonal_features:
            if 'day' in X_transformed.columns:
                day_rad = 2 * np.pi * X_transformed['day'] / 31
                X_transformed['day_sin'] = np.sin(day_rad)
                X_transformed['day_cos'] = np.cos(day_rad)
                X_transformed['is_month_end'] = (X_transformed['day'] >= 28).astype(int)
            # Month cyclical encoding (assuming month is already label encoded 0-11)
            if 'month' in X_transformed.columns:
                month_rad = 2 * np.pi * X_transformed['month'] / 12
                X_transformed['month_sin'] = np.sin(month_rad)
                X_transformed['month_cos'] = np.cos(month_rad)
        
        # 5. Demographic combinations (6 features) - use numeric encoding
        if 'job' in X_transformed.columns and 'education' in X_transformed.columns:
            X_transformed['job_education_combo'] = X_transformed['job'] * 10 + X_transformed['education']
        if 'marital' in X_transformed.columns and 'housing' in X_transformed.columns:
            X_transformed['marital_housing_combo'] = X_transformed['marital'] * 10 + X_transformed['housing']
        if 'default' in X_transformed.columns and 'loan' in X_transformed.columns:
            X_transformed['default_loan_combo'] = X_transformed['default'] * 10 + X_transformed['loan']
        if 'age' in X_transformed.columns:
            # Age group as numeric categories
            X_transformed['age_group'] = pd.cut(X_transformed['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
        if 'education' in X_transformed.columns and 'job' in X_transformed.columns:
            X_transformed['education_job_interaction'] = X_transformed['education'] * 20 + X_transformed['job']
        if 'contact' in X_transformed.columns and 'month' in X_transformed.columns:
            X_transformed['contact_month_combo'] = X_transformed['contact'] * 15 + X_transformed['month']
        
        # 6. Previous campaign features (3 features) - use numeric encoding
        if self.config.enable_previous_outcome_features:
            if 'pdays' in X_transformed.columns:
                # Use numeric binning instead of categorical labels
                X_transformed['days_since_contact_binned'] = pd.cut(X_transformed['pdays'], 
                                                                   bins=[-2, -1, 30, 180, 999], 
                                                                   labels=[0, 1, 2, 3]).astype(int)
            if 'poutcome' in X_transformed.columns:
                # Since poutcome is already label encoded, we can create indicator directly
                # Assuming 'success' has been encoded to a specific numeric value
                # We'll use the numeric values directly
                X_transformed['previous_success_indicator'] = (X_transformed['poutcome'] == 3).astype(int)  # Assuming success=3
            if 'previous' in X_transformed.columns and 'poutcome' in X_transformed.columns:
                # Create campaign history score using numeric encoding
                X_transformed['campaign_history_score'] = X_transformed['previous'] * X_transformed['poutcome']
        
        # Additional engineered features to reach 44 total
        # 7. Financial stability indicators (4 features)
        if 'balance' in X_transformed.columns and 'housing' in X_transformed.columns and 'loan' in X_transformed.columns:
            X_transformed['financial_burden'] = X_transformed['housing'].astype(int) + X_transformed['loan'].astype(int)
            X_transformed['balance_per_age'] = X_transformed['balance'] / (X_transformed['age'] + 1) if 'age' in X_transformed.columns else 0
            X_transformed['has_negative_balance'] = (X_transformed['balance'] < 0).astype(int)
            X_transformed['balance_abs_log'] = np.log1p(np.abs(X_transformed['balance']))
        
        # 8. Campaign intensity features (4 features)
        if 'campaign' in X_transformed.columns and 'previous' in X_transformed.columns:
            X_transformed['total_contacts'] = X_transformed['campaign'] + X_transformed['previous']
            X_transformed['campaign_ratio'] = X_transformed['campaign'] / (X_transformed['previous'] + 1)
            X_transformed['is_first_contact'] = (X_transformed['previous'] == 0).astype(int)
            X_transformed['high_contact_frequency'] = (X_transformed['campaign'] > 3).astype(int)
        
        # 9. Duration effectiveness features (4 features)
        if 'duration' in X_transformed.columns:
            X_transformed['duration_squared'] = X_transformed['duration'] ** 2
            X_transformed['short_call'] = (X_transformed['duration'] < 120).astype(int)  # Less than 2 minutes
            X_transformed['long_call'] = (X_transformed['duration'] > 600).astype(int)   # More than 10 minutes
            X_transformed['duration_z_score'] = (X_transformed['duration'] - X_transformed['duration'].mean()) / (X_transformed['duration'].std() + 1e-8)
        
        # 10. Contact timing features (4 features)
        if 'day' in X_transformed.columns and 'month' in X_transformed.columns:
            X_transformed['is_weekend_day'] = (X_transformed['day'] % 7 < 2).astype(int)  # Approximation
            X_transformed['is_month_start'] = (X_transformed['day'] <= 5).astype(int)
            X_transformed['day_month_interaction'] = X_transformed['day'] * X_transformed['month']
            X_transformed['optimal_timing'] = ((X_transformed['day'] >= 15) & (X_transformed['day'] <= 20)).astype(int)
        
        # 11. Demographic interaction features (4 features)  
        if 'age' in X_transformed.columns and 'marital' in X_transformed.columns:
            X_transformed['age_marital_interaction'] = X_transformed['age'] * X_transformed['marital']
        if 'education' in X_transformed.columns and 'age' in X_transformed.columns:
            X_transformed['education_age_ratio'] = X_transformed['education'] / (X_transformed['age'] + 1)
        if 'job' in X_transformed.columns and 'balance' in X_transformed.columns:
            X_transformed['job_balance_interaction'] = X_transformed['job'] * np.log1p(np.abs(X_transformed['balance']) + 1)
        if 'default' in X_transformed.columns and 'balance' in X_transformed.columns:
            X_transformed['credit_risk_score'] = X_transformed['default'] * (X_transformed['balance'] < 0).astype(int)
        
        # Fill any NaN values that might have been created during feature engineering
        X_transformed = X_transformed.fillna(0)
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
