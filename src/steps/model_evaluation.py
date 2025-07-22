"""
Model Evaluation Step for Komuniteti Predictive Maintenance Pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, Annotated, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

from zenml import step
from zenml.logger import get_logger

from ..config import MODELS_DIR
from ..schemas import ModelMetrics

logger = get_logger(__name__)

@step
def model_evaluation_step(
    trained_models: Dict[str, Any],
    best_model_metrics: ModelMetrics,
    best_model_name: str,
    processed_features: pd.DataFrame,
    target_variable: pd.Series
) -> Tuple[
    Annotated[Dict, "evaluation_report"],
    Annotated[Dict, "model_comparison"],
    Annotated[str, "evaluation_plots_path"]
]:
    """
    Evaluate trained models with comprehensive metrics and visualizations.
    
    Args:
        trained_models: Dictionary of trained models
        best_model_metrics: Metrics of the best model
        best_model_name: Name of the best model
        processed_features: Feature matrix
        target_variable: Target variable
        
    Returns:
        Tuple of evaluation report, model comparison, and plots path
    """
    
    logger.info("Starting comprehensive model evaluation")
    
    try:
        # Create test split for evaluation
        from sklearn.model_selection import train_test_split
        from ..config import model_config
        
        X_train, X_test, y_train, y_test = train_test_split(
            processed_features, 
            target_variable,
            test_size=model_config.test_size,
            stratify=target_variable,
            random_state=model_config.random_state
        )
        
        # Evaluate all models
        model_comparison = _compare_all_models(trained_models, X_test, y_test)
        
        # Detailed evaluation of best model
        evaluation_report = _detailed_model_evaluation(
            trained_models[best_model_name], 
            best_model_name,
            X_test, 
            y_test,
            processed_features.columns.tolist()
        )
        
        # Create visualizations
        plots_path = _create_evaluation_plots(
            trained_models, 
            model_comparison,
            X_test, 
            y_test,
            processed_features.columns.tolist()
        )
        
        # Business impact analysis
        evaluation_report['business_impact'] = _calculate_business_impact(
            trained_models[best_model_name], X_test, y_test
        )
        
        logger.info("Model evaluation completed successfully")
        
        return evaluation_report, model_comparison, str(plots_path)
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


def _compare_all_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Compare performance of all trained models"""
    
    logger.info("Comparing all trained models")
    
    comparison_results = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
        }
        
        comparison_results[model_name] = metrics
        
        logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    
    return comparison_results


def _detailed_model_evaluation(
    model, 
    model_name: str, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    feature_names: List[str]
) -> Dict:
    """Perform detailed evaluation of the best model"""
    
    logger.info(f"Detailed evaluation of {model_name}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, average_precision_score, classification_report, confusion_matrix
    )
    
    basic_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0,
        'avg_precision': average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    }
    
    # Confusion matrix details
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    confusion_metrics = {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
    }
    
    # Feature importance analysis
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        feature_importance = {
            'top_10_features': dict(list(zip(feature_names, importance_scores))[:10]),
            'feature_importance_stats': {
                'mean_importance': float(np.mean(importance_scores)),
                'std_importance': float(np.std(importance_scores)),
                'max_importance': float(np.max(importance_scores)),
                'min_importance': float(np.min(importance_scores))
            }
        }
    
    # Threshold analysis
    threshold_analysis = _analyze_prediction_thresholds(y_test, y_pred_proba) if y_pred_proba is not None else {}
    
    # Error analysis
    error_analysis = _analyze_prediction_errors(X_test, y_test, y_pred, feature_names)
    
    # Model stability analysis
    stability_analysis = _analyze_model_stability(model, X_test, y_test)
    
    evaluation_report = {
        'model_name': model_name,
        'basic_metrics': basic_metrics,
        'confusion_metrics': confusion_metrics,
        'feature_importance': feature_importance,
        'threshold_analysis': threshold_analysis,
        'error_analysis': error_analysis,
        'stability_analysis': stability_analysis,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return evaluation_report


def _analyze_prediction_thresholds(y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
    """Analyze model performance at different prediction thresholds"""
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_metrics = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'threshold': float(threshold),
            'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_thresh, zero_division=0)
        }
        
        threshold_metrics.append(metrics)
    
    # Find optimal threshold based on F1 score
    optimal_f1_idx = max(range(len(threshold_metrics)), key=lambda i: threshold_metrics[i]['f1_score'])
    optimal_threshold = threshold_metrics[optimal_f1_idx]['threshold']
    
    return {
        'threshold_metrics': threshold_metrics,
        'optimal_threshold': optimal_threshold,
        'optimal_f1_score': threshold_metrics[optimal_f1_idx]['f1_score']
    }


def _analyze_prediction_errors(
    X_test: pd.DataFrame, 
    y_true: pd.Series, 
    y_pred: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """Analyze prediction errors to identify patterns"""
    
    # Identify false positives and false negatives
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    error_analysis = {
        'false_positive_count': int(false_positives.sum()),
        'false_negative_count': int(false_negatives.sum()),
        'total_errors': int(false_positives.sum() + false_negatives.sum())
    }
    
    # Analyze feature patterns in errors
    if false_positives.sum() > 0:
        fp_features = X_test[false_positives].mean()
        error_analysis['false_positive_patterns'] = {
            'feature_means': fp_features.to_dict(),
            'top_fp_features': fp_features.nlargest(5).to_dict()
        }
    
    if false_negatives.sum() > 0:
        fn_features = X_test[false_negatives].mean()
        error_analysis['false_negative_patterns'] = {
            'feature_means': fn_features.to_dict(),
            'top_fn_features': fn_features.nlargest(5).to_dict()
        }
    
    return error_analysis


def _analyze_model_stability(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Analyze model stability using bootstrap sampling"""
    
    from sklearn.utils import resample
    from sklearn.metrics import f1_score, roc_auc_score
    
    n_bootstrap = 100
    f1_scores = []
    auc_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        X_boot, y_boot = resample(X_test, y_test, random_state=np.random.randint(0, 1000))
        
        # Make predictions
        y_pred_boot = model.predict(X_boot)
        
        f1_scores.append(f1_score(y_boot, y_pred_boot, zero_division=0))
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba_boot = model.predict_proba(X_boot)[:, 1]
            auc_scores.append(roc_auc_score(y_boot, y_pred_proba_boot))
    
    stability_metrics = {
        'f1_score_stability': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores)),
            'min': float(np.min(f1_scores)),
            'max': float(np.max(f1_scores)),
            'confidence_interval_95': [
                float(np.percentile(f1_scores, 2.5)),
                float(np.percentile(f1_scores, 97.5))
            ]
        }
    }
    
    if auc_scores:
        stability_metrics['auc_stability'] = {
            'mean': float(np.mean(auc_scores)),
            'std': float(np.std(auc_scores)),
            'min': float(np.min(auc_scores)),
            'max': float(np.max(auc_scores)),
            'confidence_interval_95': [
                float(np.percentile(auc_scores, 2.5)),
                float(np.percentile(auc_scores, 97.5))
            ]
        }
    
    return stability_metrics


def _calculate_business_impact(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Calculate business impact metrics"""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business impact assumptions (can be configured)
    cost_per_false_positive = 100  # Cost of unnecessary maintenance
    cost_per_false_negative = 1000  # Cost of unexpected failure
    savings_per_true_positive = 500  # Savings from prevented failure
    
    business_metrics = {
        'total_predictions': len(y_test),
        'maintenance_predicted': int(y_pred.sum()),
        'actual_maintenance_needed': int(y_test.sum()),
        'correctly_predicted_maintenance': int(tp),
        'missed_maintenance_opportunities': int(fn),
        'unnecessary_maintenance_alerts': int(fp),
        'cost_analysis': {
            'false_positive_cost': float(fp * cost_per_false_positive),
            'false_negative_cost': float(fn * cost_per_false_negative),
            'true_positive_savings': float(tp * savings_per_true_positive),
            'net_benefit': float(tp * savings_per_true_positive - fp * cost_per_false_positive - fn * cost_per_false_negative)
        },
        'efficiency_metrics': {
            'maintenance_precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'maintenance_recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'alert_accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
        }
    }
    
    return business_metrics


def _create_evaluation_plots(
    models: Dict[str, Any], 
    model_comparison: Dict,
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    feature_names: List[str]
) -> Path:
    """Create comprehensive evaluation plots"""
    
    logger.info("Creating evaluation plots")
    
    plots_dir = MODELS_DIR / "evaluation_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Model comparison plot
    _plot_model_comparison(model_comparison, plots_dir)
    
    # 2. ROC curves for all models
    _plot_roc_curves(models, X_test, y_test, plots_dir)
    
    # 3. Precision-Recall curves
    _plot_precision_recall_curves(models, X_test, y_test, plots_dir)
    
    # 4. Feature importance plot (for best model)
    best_model_name = max(model_comparison.keys(), key=lambda k: model_comparison[k]['f1_score'])
    best_model = models[best_model_name]
    _plot_feature_importance(best_model, feature_names, plots_dir)
    
    # 5. Confusion matrix
    _plot_confusion_matrix(best_model, X_test, y_test, plots_dir)
    
    # 6. Prediction distribution
    _plot_prediction_distribution(best_model, X_test, y_test, plots_dir)
    
    logger.info(f"Evaluation plots saved to {plots_dir}")
    
    return plots_dir


def _plot_model_comparison(model_comparison: Dict, plots_dir: Path):
    """Create model comparison bar chart"""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    models = list(model_comparison.keys())
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        specs=[[{"secondary_y": False}] * len(metrics)]
    )
    
    for i, metric in enumerate(metrics, 1):
        values = [model_comparison[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric, showlegend=(i==1)),
            row=1, col=i
        )
        
        fig.update_yaxis(title_text="Score", range=[0, 1], row=1, col=i)
    
    fig.update_layout(
        title="Model Performance Comparison",
        height=400,
        showlegend=False
    )
    
    fig.write_html(plots_dir / "model_comparison.html")


def _plot_roc_curves(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, plots_dir: Path):
    """Create ROC curves for all models"""
    
    fig = go.Figure()
    
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})'
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier'
    ))
    
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600, height=500
    )
    
    fig.write_html(plots_dir / "roc_curves.html")


def _plot_precision_recall_curves(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, plots_dir: Path):
    """Create Precision-Recall curves for all models"""
    
    fig = go.Figure()
    
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name} (AP = {avg_precision:.3f})'
            ))
    
    fig.update_layout(
        title="Precision-Recall Curves Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=600, height=500
    )
    
    fig.write_html(plots_dir / "precision_recall_curves.html")


def _plot_feature_importance(model, feature_names: List[str], plots_dir: Path):
    """Create feature importance plot"""
    
    if not hasattr(model, 'feature_importances_'):
        return
    
    importance_scores = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True).tail(20)  # Top 20 features
    
    fig = go.Figure(go.Bar(
        x=feature_importance_df['importance'],
        y=feature_importance_df['feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title="Top 20 Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600
    )
    
    fig.write_html(plots_dir / "feature_importance.html")


def _plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, plots_dir: Path):
    """Create confusion matrix heatmap"""
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Maintenance', 'Predicted Maintenance'],
        y=['Actual No Maintenance', 'Actual Maintenance'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        width=500, height=400
    )
    
    fig.write_html(plots_dir / "confusion_matrix.html")


def _plot_prediction_distribution(model, X_test: pd.DataFrame, y_test: pd.Series, plots_dir: Path):
    """Create prediction probability distribution plot"""
    
    if not hasattr(model, 'predict_proba'):
        return
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create histogram for each class
    fig = go.Figure()
    
    # No maintenance needed
    no_maintenance_probs = y_pred_proba[y_test == 0]
    fig.add_trace(go.Histogram(
        x=no_maintenance_probs,
        name='No Maintenance Needed',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Maintenance needed
    maintenance_probs = y_pred_proba[y_test == 1]
    fig.add_trace(go.Histogram(
        x=maintenance_probs,
        name='Maintenance Needed',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        barmode='overlay'
    )
    
    fig.write_html(plots_dir / "prediction_distribution.html") 