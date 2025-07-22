"""
Model Training Step for Komuniteti Predictive Maintenance Pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, Annotated, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime
from pathlib import Path

from zenml import step
from zenml.logger import get_logger

from ..config import model_config, MODELS_DIR
from ..schemas import ModelMetrics

logger = get_logger(__name__)

@step
def model_training_step(
    processed_features: pd.DataFrame,
    target_variable: pd.Series,
    preprocessing_artifacts: Dict[str, Any],
    feature_info: Dict
) -> Tuple[
    Annotated[Dict[str, Any], "trained_models"],
    Annotated[ModelMetrics, "best_model_metrics"],
    Annotated[str, "best_model_name"]
]:
    """
    Train multiple machine learning models and select the best one.
    
    Args:
        processed_features: Preprocessed feature matrix
        target_variable: Target variable for training
        preprocessing_artifacts: Preprocessing artifacts
        feature_info: Information about features
        
    Returns:
        Tuple of trained models, best model metrics, and best model name
    """
    
    logger.info("Starting model training")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, 
        target_variable,
        test_size=model_config.test_size,
        stratify=target_variable,
        random_state=model_config.random_state
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    logger.info(f"Positive samples in training: {y_train.sum()} ({y_train.mean():.2%})")
    
    trained_models = {}
    model_scores = {}
    
    try:
        # Train each model
        for model_name in model_config.models_to_train:
            logger.info(f"Training {model_name}")
            
            model = _get_model(model_name)
            
            # Hyperparameter tuning
            best_model = _tune_hyperparameters(model, model_name, X_train, y_train)
            
            # Train the best model
            best_model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=model_config.cv_folds, 
                scoring='f1'
            )
            
            trained_models[model_name] = best_model
            model_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"{model_name} CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['cv_mean'])
        best_model = trained_models[best_model_name]
        
        logger.info(f"Best model: {best_model_name}")
        
        # Evaluate best model on test set
        best_model_metrics = _evaluate_model(
            best_model, best_model_name, X_test, y_test, feature_info['feature_names']
        )
        
        # Save the best model
        _save_best_model(best_model, best_model_name, best_model_metrics)
        
        logger.info("Model training completed successfully")
        
        return trained_models, best_model_metrics, best_model_name
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


def _get_model(model_name: str):
    """Get model instance based on name"""
    
    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=model_config.random_state,
            n_jobs=-1
        )
    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            random_state=model_config.random_state,
            eval_metric='logloss'
        )
    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(
            random_state=model_config.random_state,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_hyperparameter_grid(model_name: str) -> Dict:
    """Get hyperparameter grid for model tuning"""
    
    if model_name == "random_forest":
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
    elif model_name == "xgboost":
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, 2, 3]  # For imbalanced classes
        }
    elif model_name == "lightgbm":
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'class_weight': ['balanced', None]
        }
    else:
        return {}


def _tune_hyperparameters(model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Tune hyperparameters using GridSearchCV"""
    
    logger.info(f"Tuning hyperparameters for {model_name}")
    
    param_grid = _get_hyperparameter_grid(model_name)
    
    if not param_grid:
        logger.warning(f"No hyperparameter grid defined for {model_name}, using default parameters")
        return model
    
    # Use a smaller grid for faster training if dataset is large
    if len(X_train) > 10000:
        # Reduce grid size for large datasets
        reduced_grid = {}
        for key, values in param_grid.items():
            reduced_grid[key] = values[:2] if len(values) > 2 else values
        param_grid = reduced_grid
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,  # Use fewer folds for hyperparameter tuning
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def _evaluate_model(
    model, 
    model_name: str, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    feature_names: list
) -> ModelMetrics:
    """Evaluate model performance and create metrics"""
    
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC (if probabilities available)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
    
    # Feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores.tolist()))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Log detailed results
    logger.info(f"{model_name} Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    class_report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Create model metrics
    model_metrics = ModelMetrics(
        model_name=model_name,
        model_version="1.0.0",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        training_date=datetime.now(),
        training_samples=len(X_test),  # Using test set size as proxy
        feature_importance=feature_importance
    )
    
    return model_metrics


def _save_best_model(model, model_name: str, metrics: ModelMetrics) -> None:
    """Save the best model and its metadata"""
    
    logger.info(f"Saving best model: {model_name}")
    
    # Save model
    model_path = MODELS_DIR / f"best_model_{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'model_path': str(model_path),
        'metrics': metrics.dict(),
        'saved_at': datetime.now().isoformat()
    }
    
    metadata_path = MODELS_DIR / "best_model_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")


# Utility functions for model loading and prediction

def load_best_model() -> Tuple[Any, Dict]:
    """Load the best trained model and its metadata"""
    
    metadata_path = MODELS_DIR / "best_model_metadata.joblib"
    
    if not metadata_path.exists():
        raise FileNotFoundError("No trained model found. Please run training first.")
    
    metadata = joblib.load(metadata_path)
    model_path = metadata['model_path']
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    return model, metadata


def predict_maintenance_probability(
    model,
    features: pd.DataFrame,
    return_probabilities: bool = True
) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        features: Feature matrix
        return_probabilities: Whether to return probabilities or binary predictions
        
    Returns:
        Predictions or probabilities
    """
    
    if return_probabilities and hasattr(model, 'predict_proba'):
        # Return probability of positive class (maintenance needed)
        return model.predict_proba(features)[:, 1]
    else:
        # Return binary predictions
        return model.predict(features)


def get_feature_importance(model, feature_names: list) -> Dict[str, float]:
    """Get feature importance from trained model"""
    
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores.tolist()))
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    else:
        return {}


def categorize_risk(probability: float, 
                   low_threshold: float = 0.3, 
                   high_threshold: float = 0.7) -> str:
    """Categorize maintenance risk based on probability"""
    
    if probability < low_threshold:
        return "low"
    elif probability < high_threshold:
        return "medium"
    else:
        return "high" 