import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from bank_marketing_term_deposit_prediction.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        """Create the model based on configuration"""
        model_type = self.config.model_type
        params = self.config.model_params
        
        if model_type == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params)
        elif model_type == "extra_trees":
            return ExtraTreesClassifier(**params)
        elif model_type == "logistic_regression":
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        """
        Fit the classifier to the training data with optional sample weights for cost-sensitive learning.

        Parameters:
        X: Training features.
        y: Target labels (pandas Series or numpy array).
        sample_weight: Optional sample weights for cost-sensitive learning.

        Returns:
        self: Fitted classifier.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        # Accept both pandas Series and numpy arrays
        if not (isinstance(y, pd.Series) or hasattr(y, '__array__')):
            raise ValueError("y must be a pandas Series or numpy array")
        
        # Apply sample weights for cost-sensitive learning if provided
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Save the model wrapper as an artifact

        Parameters:
        path (str): The file path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """
        Load the model wrapper from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)