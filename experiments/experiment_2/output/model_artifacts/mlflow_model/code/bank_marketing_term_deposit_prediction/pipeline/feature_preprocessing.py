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

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
            
        X_transformed = X.copy()
        
        # Duration-based features according to experiment plan
        if 'duration' in X_transformed.columns:
            X_transformed['duration_log'] = np.log1p(X_transformed['duration'])
            X_transformed['duration_sqrt'] = np.sqrt(X_transformed['duration'])
        
        # Balance-based features according to experiment plan
        if 'balance' in X_transformed.columns:
            X_transformed['balance_log'] = np.log1p(X_transformed['balance'] - self.balance_min + 1)
            X_transformed['balance_positive'] = (X_transformed['balance'] > 0).astype(int)
        
        # Campaign features according to experiment plan
        if 'campaign' in X_transformed.columns and 'previous' in X_transformed.columns:
            X_transformed['campaign_per_previous'] = X_transformed['campaign'] / (X_transformed['previous'] + 1)
        
        if 'pdays' in X_transformed.columns:
            X_transformed['pdays_binary'] = (X_transformed['pdays'] != -1).astype(int)
        
        # Temporal features - day cyclical encoding
        if 'day' in X_transformed.columns:
            day_rad = 2 * np.pi * X_transformed['day'] / 31
            X_transformed['day_sin'] = np.sin(day_rad)
            X_transformed['day_cos'] = np.cos(day_rad)
            X_transformed['is_month_end'] = (X_transformed['day'] >= 28).astype(int)
        
        # Interaction features according to experiment plan
        if 'age' in X_transformed.columns and 'balance' in X_transformed.columns:
            X_transformed['age_balance_ratio'] = X_transformed['age'] / (np.abs(X_transformed['balance']) + 1)
            
        if 'duration' in X_transformed.columns and 'campaign' in X_transformed.columns:
            X_transformed['duration_campaign_ratio'] = X_transformed['duration'] / (X_transformed['campaign'] + 1)
        
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
