from typing import Optional
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from bank_marketing_term_deposit_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        # Column mapping from V1-V16 to meaningful names
        self.column_mapping = {
            'V1': 'age', 'V2': 'job', 'V3': 'marital', 'V4': 'education', 
            'V5': 'default', 'V6': 'balance', 'V7': 'housing', 'V8': 'loan',
            'V9': 'contact', 'V10': 'day', 'V11': 'month', 'V12': 'duration',
            'V13': 'campaign', 'V14': 'pdays', 'V15': 'previous', 'V16': 'poutcome',
            'target': 'y'
        }
        self.label_encoders = {}
        self.dummy_columns = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        X_copy = X.copy()
        
        # Apply column mapping
        X_copy = self._apply_column_mapping(X_copy)
        
        # Convert target variable from 1/2 to 0/1 if present
        if 'y' in X_copy.columns:
            X_copy['y'] = X_copy['y'] - 1
        
        # Fit label encoders for specified categorical columns
        label_encode_cols = self.config.label_encode_columns
        for col in label_encode_cols:
            if col in X_copy.columns:
                le = LabelEncoder()
                le.fit(X_copy[col].astype(str))
                self.label_encoders[col] = le
        
        # Store one-hot encoding column info by doing a dummy fit
        onehot_cols = self.config.onehot_encode_columns
        for col in onehot_cols:
            if col in X_copy.columns:
                dummies = pd.get_dummies(X_copy[col], prefix=col, drop_first=False)
                self.dummy_columns[col] = list(dummies.columns)
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
            
        X_transformed = X.copy()
        
        # Apply column mapping
        X_transformed = self._apply_column_mapping(X_transformed)
        
        # Convert target variable from 1/2 to 0/1 if present
        if 'y' in X_transformed.columns:
            X_transformed['y'] = X_transformed['y'] - 1
        
        # Apply one-hot encoding for specified columns with consistent columns
        onehot_cols = self.config.onehot_encode_columns
        for col in onehot_cols:
            if col in X_transformed.columns:
                dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=False)
                X_transformed = X_transformed.drop(columns=[col])
                
                # Ensure consistent columns from training
                expected_columns = self.dummy_columns[col]
                
                # Add missing columns with zeros
                for expected_col in expected_columns:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                
                # Remove unexpected columns (not seen during training)
                for dummy_col in list(dummies.columns):
                    if dummy_col not in expected_columns:
                        dummies = dummies.drop(columns=[dummy_col])
                
                # Reorder to match training order
                dummies = dummies[expected_columns]
                
                X_transformed = pd.concat([X_transformed, dummies], axis=1)
        
        # Apply label encoding for specified columns
        label_encode_cols = self.config.label_encode_columns
        for col in label_encode_cols:
            if col in X_transformed.columns and col in self.label_encoders:
                # Handle unknown categories by mapping them to the first class
                col_values = X_transformed[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                
                # Replace unknown values with the first known class
                unknown_mask = ~col_values.isin(known_classes)
                if unknown_mask.any():
                    col_values.loc[unknown_mask] = self.label_encoders[col].classes_[0]
                
                X_transformed[col] = self.label_encoders[col].transform(col_values)
        
        return X_transformed

    def _apply_column_mapping(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping from V1, V2, etc. to meaningful names"""
        # Only rename columns that exist in the dataframe
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in X.columns}
        return X.rename(columns=rename_dict)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
