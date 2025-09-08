import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path
import os

from bank_marketing_term_deposit_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, model, X_train, y_train, X_test, y_test, output_dir):
        """
        Comprehensive model evaluation with plots and metrics
        
        Parameters:
        model: Fitted model
        X_train, y_train: Training data
        X_test, y_test: Test data
        output_dir: Directory to save plots
        
        Returns:
        dict: Evaluation metrics and results
        """
        # Create output directory if it doesn't exist
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate primary metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Cross-validation scores
        cv_scores = self._cross_validate_model(model, X_train, y_train)
        
        # Create evaluation plots
        self._create_roc_curve(y_test, y_pred_proba, plots_dir, roc_auc)
        self._create_precision_recall_curve(y_test, y_pred_proba, plots_dir, avg_precision)
        self._create_confusion_matrix_plot(y_test, y_pred, plots_dir)
        self._create_calibration_plot(y_test, y_pred_proba, plots_dir)
        self._create_feature_importance_plot(model, X_train.columns, plots_dir)
        self._create_prediction_distribution_plot(y_pred_proba, y_test, plots_dir)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": class_report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

    def _cross_validate_model(self, model, X, y):
        """Perform cross-validation"""
        if self.config.stratified:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        else:
            cv = self.config.cv_folds
            
        if self.config.primary_metric == "roc_auc":
            scoring = "roc_auc"
        else:
            scoring = self.config.primary_metric
            
        return cross_val_score(model.model, X, y, cv=cv, scoring=scoring)

    def _create_roc_curve(self, y_true, y_pred_proba, output_dir, auc_score):
        """Create ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color=self.app_color_palette[0], width=3)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color=self.app_color_palette[1], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - Bank Marketing Model',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        fig.write_html(output_dir / "roc_curve.html", include_plotlyjs=True, 
                      config={'responsive': True, 'displayModeBar': False})

    def _create_precision_recall_curve(self, y_true, y_pred_proba, output_dir, avg_precision):
        """Create Precision-Recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color=self.app_color_palette[2], width=3)
        ))
        
        # Add baseline
        baseline = np.mean(y_true)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Baseline (AP = {baseline:.3f})',
            line=dict(color=self.app_color_palette[1], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve - Bank Marketing Model',
            xaxis_title='Recall',
            yaxis_title='Precision',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        fig.write_html(output_dir / "precision_recall_curve.html", include_plotlyjs=True,
                      config={'responsive': True, 'displayModeBar': False})

    def _create_confusion_matrix_plot(self, y_true, y_pred, output_dir):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            colorscale='Purples',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title='Confusion Matrix - Bank Marketing Model',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            )
        )
        
        fig.write_html(output_dir / "confusion_matrix.html", include_plotlyjs=True,
                      config={'responsive': True, 'displayModeBar': False})

    def _create_calibration_plot(self, y_true, y_pred_proba, output_dir):
        """Create calibration plot"""
        fraction_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color=self.app_color_palette[1], width=2, dash='dash')
        ))
        
        # Model calibration
        fig.add_trace(go.Scatter(
            x=mean_predicted_value, y=fraction_positives,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color=self.app_color_palette[0], width=3),
            marker=dict(size=8, color=self.app_color_palette[0])
        ))
        
        fig.update_layout(
            title='Calibration Plot - Bank Marketing Model',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        fig.write_html(output_dir / "calibration_plot.html", include_plotlyjs=True,
                      config={'responsive': True, 'displayModeBar': False})

    def _create_feature_importance_plot(self, model, feature_names, output_dir):
        """Create feature importance plot"""
        if hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            
            # Get top 20 features
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            
            fig = go.Figure([go.Bar(
                x=feature_importance_df['importance'],
                y=feature_importance_df['feature'],
                orientation='h',
                marker_color=self.app_color_palette[3]
            )])
            
            fig.update_layout(
                title='Top 20 Feature Importances - Bank Marketing Model',
                xaxis_title='Importance',
                yaxis_title='Features',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8B5CF6', size=12),
                title_font=dict(color='#7C3AED', size=16),
                xaxis=dict(
                    gridcolor='rgba(139,92,246,0.2)',
                    zerolinecolor='rgba(139,92,246,0.3)',
                    tickfont=dict(color='#8B5CF6', size=11),
                    title_font=dict(color='#7C3AED', size=12)
                ),
                yaxis=dict(
                    tickfont=dict(color='#8B5CF6', size=11),
                    title_font=dict(color='#7C3AED', size=12)
                ),
                height=600
            )
            
            fig.write_html(output_dir / "feature_importance.html", include_plotlyjs=True,
                          config={'responsive': True, 'displayModeBar': False})

    def _create_prediction_distribution_plot(self, y_pred_proba, y_true, output_dir):
        """Create prediction distribution plot"""
        # Create histogram by class
        class_0_probs = y_pred_proba[y_true == 0]
        class_1_probs = y_pred_proba[y_true == 1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=class_0_probs,
            name='No Subscription (Class 0)',
            opacity=0.7,
            nbinsx=30,
            marker_color=self.app_color_palette[1]
        ))
        
        fig.add_trace(go.Histogram(
            x=class_1_probs,
            name='Subscription (Class 1)',
            opacity=0.7,
            nbinsx=30,
            marker_color=self.app_color_palette[0]
        ))
        
        fig.update_layout(
            title='Prediction Probability Distribution by Class',
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        fig.write_html(output_dir / "prediction_distribution.html", include_plotlyjs=True,
                      config={'responsive': True, 'displayModeBar': False})
