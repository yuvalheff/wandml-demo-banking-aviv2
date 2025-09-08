import pandas as pd
import numpy as np
import mlflow
import sklearn
from pathlib import Path
import os
import json
from imblearn.over_sampling import SMOTE

from bank_marketing_term_deposit_prediction.pipeline.feature_preprocessing import FeatureProcessor
from bank_marketing_term_deposit_prediction.pipeline.data_preprocessing import DataProcessor
from bank_marketing_term_deposit_prediction.pipeline.model import ModelWrapper
from bank_marketing_term_deposit_prediction.config import Config
from experiment_scripts.evaluation import ModelEvaluator
from bank_marketing_term_deposit_prediction.model_pipeline import ModelPipeline

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        """
        Run the complete ML experiment pipeline.
        
        Parameters:
        train_dataset_path (str): Path to training data CSV
        test_dataset_path (str): Path to test data CSV  
        output_dir (str): Output directory for artifacts and results
        seed (int): Random seed for reproducibility
        
        Returns:
        dict: Experiment results with required format
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create output directories
        output_path = Path(output_dir)
        artifacts_dir = output_path / "output" / "model_artifacts"
        general_artifacts_dir = output_path / "output" / "general_artifacts"
        
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        general_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Load data
            print("🔄 Loading training and test datasets...")
            train_df = pd.read_csv(train_dataset_path)
            test_df = pd.read_csv(test_dataset_path)
            
            print(f"✅ Loaded {len(train_df)} training samples and {len(test_df)} test samples")
            
            # 2. Initialize pipeline components
            print("🔧 Initializing pipeline components...")
            data_processor = DataProcessor(self._config.data_prep)
            feature_processor = FeatureProcessor(self._config.feature_prep)
            model_wrapper = ModelWrapper(self._config.model)
            
            # 3. Fit data processor on training data
            print("🔄 Fitting data processor...")
            data_processor.fit(train_df)
            
            # Transform training data
            train_processed = data_processor.transform(train_df)
            print(f"✅ Data processing complete. Shape: {train_processed.shape}")
            
            # Separate features and target
            y_train = train_processed['y']
            X_train = train_processed.drop(columns=['y'])
            
            # 4. Fit feature processor and transform training data
            print("🔄 Fitting feature processor...")
            feature_processor.fit(X_train, y_train)
            X_train_features = feature_processor.transform(X_train)
            print(f"✅ Feature engineering complete. Shape: {X_train_features.shape}")
            
            # 5. Apply SMOTE oversampling to balance training data
            print("⚖️ Applying SMOTE oversampling...")
            print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
            
            smote = SMOTE(k_neighbors=5, random_state=seed)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_features, y_train)
            
            print(f"After SMOTE - Class distribution: {np.bincount(y_train_smote)}")
            print(f"Training data shape after SMOTE: {X_train_smote.shape}")
            
            # 6. Train model on SMOTE-enhanced data
            print("🔄 Training model on SMOTE-enhanced data...")
            model_wrapper.fit(X_train_smote, y_train_smote)
            print("✅ Model training complete")
            
            # 7. Process test data
            print("🔄 Processing test data...")
            test_processed = data_processor.transform(test_df)
            y_test = test_processed['y']
            X_test = test_processed.drop(columns=['y'])
            X_test_features = feature_processor.transform(X_test)
            print(f"✅ Test data processing complete. Shape: {X_test_features.shape}")
            
            # 8. Evaluate model (use original training data for CV to avoid data leakage)
            print("📊 Evaluating model performance...")
            evaluator = ModelEvaluator(self._config.model_evaluation)
            eval_results = evaluator.evaluate_model(
                model_wrapper, X_train_features, y_train, X_test_features, y_test, output_path / "output"
            )
            
            print(f"✅ Model evaluation complete. ROC-AUC: {eval_results['roc_auc']:.4f}")
            
            # 9. Save individual model artifacts 
            print("💾 Saving model artifacts...")
            data_processor.save(artifacts_dir / "data_processor.pkl")
            feature_processor.save(artifacts_dir / "feature_processor.pkl")
            model_wrapper.save(artifacts_dir / "trained_models.pkl")
            
            # 10. Create and test ModelPipeline
            print("🔧 Creating ModelPipeline...")
            pipeline = ModelPipeline(data_processor, feature_processor, model_wrapper)
            
            # Test pipeline with sample data
            sample_data = test_df.head(3).drop(columns=['y'] if 'y' in test_df.columns else [])
            sample_predictions = pipeline.predict(sample_data)
            sample_probabilities = pipeline.predict_proba(sample_data)
            print(f"✅ Pipeline test successful. Sample predictions: {sample_predictions}")
            
            # 11. Save and conditionally log MLflow model
            print("🚀 Saving MLflow model...")
            output_model_path = artifacts_dir / "mlflow_model"
            relative_path_for_return = "output/model_artifacts/mlflow_model/"
            
            # Prepare sample input for signature
            sample_input = sample_data.head(1)
            signature = mlflow.models.infer_signature(sample_input, pipeline.predict(sample_input))
            
            # Always save the model locally for harness validation
            print(f"💾 Saving model to local disk: {output_model_path}")
            mlflow.sklearn.save_model(
                pipeline,
                path=str(output_model_path),
                code_paths=["bank_marketing_term_deposit_prediction"],  # Bundle custom code
                signature=signature
            )
            
            # Conditionally log to MLflow if run ID is provided
            active_run_id = "55d5aadc00d34e5abbe22df6e0c68855"
            logged_model_uri = None
            
            if active_run_id and active_run_id != 'None' and active_run_id.strip():
                print(f"✅ Active MLflow run ID '{active_run_id}' detected. Logging model...")
                try:
                    with mlflow.start_run(run_id=active_run_id):
                        logged_model_info = mlflow.sklearn.log_model(
                            pipeline,
                            artifact_path="model",
                            code_paths=["bank_marketing_term_deposit_prediction"],
                            signature=signature
                        )
                        logged_model_uri = logged_model_info.model_uri
                        print(f"✅ Model logged to MLflow: {logged_model_uri}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not log to MLflow: {e}")
                    logged_model_uri = None
            else:
                print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
            
            # 12. Prepare return results
            model_artifacts = [
                "data_processor.pkl",
                "feature_processor.pkl", 
                "trained_models.pkl"
            ]
            
            # Create input example for the return
            input_example = sample_input.to_dict('records')[0]
            
            mlflow_model_info = {
                "model_path": relative_path_for_return,
                "logged_model_uri": logged_model_uri,
                "model_type": "sklearn",
                "task_type": "classification",
                "signature": {
                    "inputs": signature.inputs.to_dict() if signature.inputs else None,
                    "outputs": signature.outputs.to_dict() if signature.outputs else None
                },
                "input_example": input_example,
                "python_model_class": "ModelPipeline",
                "framework_version": sklearn.__version__
            }
            
            print("✅ Experiment completed successfully!")
            
            return {
                "metric_name": "roc_auc",
                "metric_value": eval_results['roc_auc'],
                "model_artifacts": model_artifacts,
                "mlflow_model_info": mlflow_model_info
            }
            
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            raise e