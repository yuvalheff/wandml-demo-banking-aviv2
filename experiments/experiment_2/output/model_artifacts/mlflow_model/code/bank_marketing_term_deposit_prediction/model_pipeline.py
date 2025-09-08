"""
Bank Marketing Term Deposit Prediction Pipeline

A complete ML pipeline that combines data processing, feature engineering,
and model prediction for bank marketing term deposit classification.
"""

import pandas as pd
import numpy as np
from typing import Union

from bank_marketing_term_deposit_prediction.pipeline.data_preprocessing import DataProcessor
from bank_marketing_term_deposit_prediction.pipeline.feature_preprocessing import FeatureProcessor
from bank_marketing_term_deposit_prediction.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete pipeline for bank marketing term deposit prediction.
    
    This pipeline handles the complete workflow from raw input data
    to final predictions, including data preprocessing, feature engineering,
    and model prediction.
    """
    
    def __init__(self, data_processor=None, feature_processor=None, model=None):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: Fitted DataProcessor instance
        feature_processor: Fitted FeatureProcessor instance  
        model: Fitted ModelWrapper instance
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
        self.is_fitted = False
        
        # Check if all components are provided and fitted
        if (self.data_processor is not None and 
            self.feature_processor is not None and 
            self.model is not None and
            hasattr(self.data_processor, 'is_fitted') and self.data_processor.is_fitted and
            hasattr(self.feature_processor, 'is_fitted') and self.feature_processor.is_fitted and
            hasattr(self.model, 'is_fitted') and self.model.is_fitted):
            self.is_fitted = True
    
    def predict(self, X: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Parameters:
        X: Input data - can be DataFrame, dict, or list of dicts
        
        Returns:
        np.ndarray: Predicted class labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be DataFrame, dict, or list of dicts")
        
        # Apply data processing
        X_processed = self.data_processor.transform(X)
        
        # Remove target column if present (for prediction on new data)
        if 'y' in X_processed.columns:
            X_processed = X_processed.drop(columns=['y'])
        
        # Apply feature engineering
        X_features = self.feature_processor.transform(X_processed)
        
        # Make predictions
        predictions = self.model.predict(X_features)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Parameters:
        X: Input data - can be DataFrame, dict, or list of dicts
        
        Returns:
        np.ndarray: Predicted class probabilities [prob_class_0, prob_class_1]
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Convert input to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be DataFrame, dict, or list of dicts")
        
        # Apply data processing
        X_processed = self.data_processor.transform(X)
        
        # Remove target column if present (for prediction on new data)
        if 'y' in X_processed.columns:
            X_processed = X_processed.drop(columns=['y'])
        
        # Apply feature engineering
        X_features = self.feature_processor.transform(X_processed)
        
        # Make probability predictions
        probabilities = self.model.predict_proba(X_features)
        
        return probabilities
    
    def get_feature_names(self):
        """
        Get the names of features used by the model.
        
        Returns:
        list: Feature names after all processing steps
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted to get feature names")
        
        # This would need to be implemented based on the actual feature names
        # after all transformations. For now, return None.
        return None
