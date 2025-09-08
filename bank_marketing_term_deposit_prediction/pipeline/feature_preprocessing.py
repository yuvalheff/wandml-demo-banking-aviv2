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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # No fitting required for the feature engineering steps defined
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
            
        X_transformed = X.copy()
        
        # 1. Duration-based features
        if 'duration' in X_transformed.columns:
            X_transformed['duration_log'] = np.log1p(X_transformed['duration'])
            X_transformed['duration_short'] = (X_transformed['duration'] < 100).astype(int)
            X_transformed['duration_long'] = (X_transformed['duration'] > 500).astype(int)
        
        # 2. Balance-based features
        if 'balance' in X_transformed.columns:
            X_transformed['balance_log'] = X_transformed['balance'].apply(
                lambda x: np.log(x) if x > 0 else 0
            )
            X_transformed['balance_negative'] = (X_transformed['balance'] < 0).astype(int)
            X_transformed['balance_zero'] = (X_transformed['balance'] == 0).astype(int)
        
        # 3. Campaign features
        if 'campaign' in X_transformed.columns:
            X_transformed['campaign_multiple'] = (X_transformed['campaign'] > 1).astype(int)
        
        if 'pdays' in X_transformed.columns:
            X_transformed['pdays_contacted_before'] = (X_transformed['pdays'] != -1).astype(int)
        
        # 4. Seasonal features
        if 'month' in X_transformed.columns:
            month_high_success = ['mar', 'dec', 'sep']
            month_low_success = ['may', 'jul', 'jun']
            
            X_transformed['month_high_success'] = X_transformed['month'].isin(month_high_success).astype(int)
            X_transformed['month_low_success'] = X_transformed['month'].isin(month_low_success).astype(int)
        
        # 5. Previous outcome features
        if 'poutcome' in X_transformed.columns:
            X_transformed['poutcome_success'] = (X_transformed['poutcome'] == 'success').astype(int)
            X_transformed['poutcome_unknown'] = (X_transformed['poutcome'] == 'unknown').astype(int)
        
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
