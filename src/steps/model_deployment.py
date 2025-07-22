"""
Model Deployment Step for Komuniteti Predictive Maintenance Pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, Annotated, Dict, Any
import joblib
import json
from datetime import datetime
from pathlib import Path

from zenml import step
from zenml.logger import get_logger

from ..config import MODELS_DIR, api_config
from ..schemas import ModelMetrics

logger = get_logger(__name__)

@step
def model_deployment_step(
    trained_models: Dict[str, Any],
    best_model_metrics: ModelMetrics,
    best_model_name: str,
    preprocessing_artifacts: Dict[str, Any],
    feature_info: Dict
) -> Tuple[
    Annotated[str, "deployment_status"],
    Annotated[Dict, "deployment_info"],
    Annotated[str, "model_version"]
]:
    """
    Deploy the best model for serving predictions.
    
    Args:
        trained_models: Dictionary of trained models
        best_model_metrics: Metrics of the best model
        best_model_name: Name of the best model
        preprocessing_artifacts: Preprocessing artifacts
        feature_info: Feature information
        
    Returns:
        Tuple of deployment status, deployment info, and model version
    """
    
    logger.info(f"Starting model deployment for {best_model_name}")
    
    try:
        # Generate model version
        model_version = _generate_model_version()
        
        # Prepare deployment package
        deployment_package = _prepare_deployment_package(
            trained_models[best_model_name],
            best_model_name,
            best_model_metrics,
            preprocessing_artifacts,
            feature_info,
            model_version
        )
        
        # Save deployment package
        deployment_info = _save_deployment_package(deployment_package, model_version)
        
        # Create model registry entry
        _register_model(best_model_name, model_version, best_model_metrics, deployment_info)
        
        # Validate deployment
        validation_result = _validate_deployment(deployment_info)
        
        if validation_result['status'] == 'success':
            deployment_status = "deployed"
            logger.info(f"Model {best_model_name} v{model_version} deployed successfully")
        else:
            deployment_status = "failed"
            logger.error(f"Deployment validation failed: {validation_result['error']}")
        
        return deployment_status, deployment_info, model_version
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise


def _generate_model_version() -> str:
    """Generate a version string for the model"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"v{timestamp}"


def _prepare_deployment_package(
    model: Any,
    model_name: str,
    metrics: ModelMetrics,
    preprocessing_artifacts: Dict[str, Any],
    feature_info: Dict,
    version: str
) -> Dict[str, Any]:
    """Prepare a complete deployment package"""
    
    logger.info("Preparing deployment package")
    
    deployment_package = {
        'model': model,
        'model_name': model_name,
        'model_version': version,
        'metrics': metrics.dict(),
        'preprocessing_artifacts': preprocessing_artifacts,
        'feature_info': feature_info,
        'deployment_config': {
            'prediction_threshold': api_config.prediction_threshold,
            'low_risk_threshold': api_config.low_risk_threshold,
            'high_risk_threshold': api_config.high_risk_threshold,
            'api_version': "1.0.0",
            'supported_formats': ['json'],
            'max_batch_size': 1000
        },
        'metadata': {
            'deployed_at': datetime.now().isoformat(),
            'framework_version': _get_framework_versions(),
            'feature_names': feature_info['feature_names'],
            'model_type': type(model).__name__,
            'training_samples': metrics.training_samples
        }
    }
    
    return deployment_package


def _get_framework_versions() -> Dict[str, str]:
    """Get versions of key frameworks"""
    
    versions = {}
    
    try:
        import sklearn
        versions['scikit-learn'] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import xgboost
        versions['xgboost'] = xgboost.__version__
    except ImportError:
        pass
    
    try:
        import lightgbm
        versions['lightgbm'] = lightgbm.__version__
    except ImportError:
        pass
    
    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except ImportError:
        pass
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        pass
    
    return versions


def _save_deployment_package(package: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Save the deployment package to disk"""
    
    logger.info("Saving deployment package")
    
    # Create deployment directory
    deployment_dir = MODELS_DIR / "deployments" / version
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = deployment_dir / "model.joblib"
    joblib.dump(package['model'], model_path)
    
    # Save preprocessing artifacts
    preprocessing_path = deployment_dir / "preprocessing.joblib"
    joblib.dump(package['preprocessing_artifacts'], preprocessing_path)
    
    # Save metadata and configuration
    metadata_path = deployment_dir / "metadata.json"
    metadata = {
        'model_name': package['model_name'],
        'model_version': package['model_version'],
        'metrics': package['metrics'],
        'feature_info': package['feature_info'],
        'deployment_config': package['deployment_config'],
        'metadata': package['metadata']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Create deployment info
    deployment_info = {
        'deployment_dir': str(deployment_dir),
        'model_path': str(model_path),
        'preprocessing_path': str(preprocessing_path),
        'metadata_path': str(metadata_path),
        'deployment_config': package['deployment_config'],
        'feature_names': package['feature_info']['feature_names'],
        'model_version': version,
        'deployed_at': package['metadata']['deployed_at']
    }
    
    # Save deployment info for easy access
    deployment_info_path = MODELS_DIR / "current_deployment.json"
    with open(deployment_info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2, default=str)
    
    logger.info(f"Deployment package saved to {deployment_dir}")
    
    return deployment_info


def _register_model(
    model_name: str, 
    version: str, 
    metrics: ModelMetrics, 
    deployment_info: Dict[str, Any]
) -> None:
    """Register the model in the model registry"""
    
    logger.info("Registering model in registry")
    
    # Create model registry directory
    registry_dir = MODELS_DIR / "registry"
    registry_dir.mkdir(exist_ok=True)
    
    # Load existing registry or create new one
    registry_file = registry_dir / "model_registry.json"
    
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'models': {}}
    
    # Add new model entry
    if model_name not in registry['models']:
        registry['models'][model_name] = {'versions': []}
    
    # Add version entry
    version_entry = {
        'version': version,
        'deployed_at': deployment_info['deployed_at'],
        'deployment_path': deployment_info['deployment_dir'],
        'metrics': metrics.dict(),
        'status': 'active',
        'performance': {
            'f1_score': metrics.f1_score,
            'roc_auc': metrics.roc_auc,
            'precision': metrics.precision,
            'recall': metrics.recall
        }
    }
    
    registry['models'][model_name]['versions'].append(version_entry)
    
    # Mark this version as current
    registry['models'][model_name]['current_version'] = version
    registry['last_updated'] = datetime.now().isoformat()
    
    # Save updated registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    
    logger.info(f"Model {model_name} v{version} registered successfully")


def _validate_deployment(deployment_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the deployment package"""
    
    logger.info("Validating deployment")
    
    try:
        # Check if all required files exist
        required_files = [
            deployment_info['model_path'],
            deployment_info['preprocessing_path'],
            deployment_info['metadata_path']
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                return {
                    'status': 'failed',
                    'error': f"Required file missing: {file_path}"
                }
        
        # Try loading the model
        model = joblib.load(deployment_info['model_path'])
        preprocessing = joblib.load(deployment_info['preprocessing_path'])
        
        # Load metadata
        with open(deployment_info['metadata_path'], 'r') as f:
            metadata = json.load(f)
        
        # Validate model can make predictions (with dummy data)
        n_features = len(deployment_info['feature_names'])
        dummy_data = pd.DataFrame(np.random.randn(1, n_features), columns=deployment_info['feature_names'])
        
        # Test prediction
        prediction = model.predict(dummy_data)
        probability = model.predict_proba(dummy_data) if hasattr(model, 'predict_proba') else None
        
        validation_result = {
            'status': 'success',
            'checks_passed': [
                'all_files_exist',
                'model_loadable',
                'preprocessing_loadable',
                'metadata_readable',
                'prediction_works'
            ],
            'model_type': type(model).__name__,
            'feature_count': n_features,
            'supports_probabilities': probability is not None
        }
        
        logger.info("Deployment validation successful")
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


# Utility functions for loading deployed models

def load_current_deployment() -> Tuple[Any, Dict[str, Any]]:
    """Load the currently deployed model and its metadata"""
    
    deployment_info_path = MODELS_DIR / "current_deployment.json"
    
    if not deployment_info_path.exists():
        raise FileNotFoundError("No deployment found. Please deploy a model first.")
    
    with open(deployment_info_path, 'r') as f:
        deployment_info = json.load(f)
    
    # Load model and preprocessing
    model = joblib.load(deployment_info['model_path'])
    preprocessing = joblib.load(deployment_info['preprocessing_path'])
    
    # Load metadata
    with open(deployment_info['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    return model, {
        'preprocessing': preprocessing,
        'metadata': metadata,
        'deployment_info': deployment_info
    }


def get_model_registry() -> Dict[str, Any]:
    """Get the model registry information"""
    
    registry_file = MODELS_DIR / "registry" / "model_registry.json"
    
    if not registry_file.exists():
        return {'models': {}}
    
    with open(registry_file, 'r') as f:
        return json.load(f)


def load_specific_model_version(model_name: str, version: str) -> Tuple[Any, Dict[str, Any]]:
    """Load a specific model version"""
    
    registry = get_model_registry()
    
    if model_name not in registry['models']:
        raise ValueError(f"Model {model_name} not found in registry")
    
    # Find the version
    model_versions = registry['models'][model_name]['versions']
    version_info = None
    
    for v in model_versions:
        if v['version'] == version:
            version_info = v
            break
    
    if version_info is None:
        raise ValueError(f"Version {version} not found for model {model_name}")
    
    # Load the model
    deployment_path = Path(version_info['deployment_path'])
    model = joblib.load(deployment_path / "model.joblib")
    preprocessing = joblib.load(deployment_path / "preprocessing.joblib")
    
    with open(deployment_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return model, {
        'preprocessing': preprocessing,
        'metadata': metadata,
        'version_info': version_info
    }


def rollback_to_previous_version(model_name: str) -> Dict[str, Any]:
    """Rollback to the previous model version"""
    
    registry = get_model_registry()
    
    if model_name not in registry['models']:
        raise ValueError(f"Model {model_name} not found in registry")
    
    versions = registry['models'][model_name]['versions']
    current_version = registry['models'][model_name]['current_version']
    
    # Find current version index
    current_idx = None
    for i, v in enumerate(versions):
        if v['version'] == current_version:
            current_idx = i
            break
    
    if current_idx is None or current_idx == 0:
        raise ValueError("No previous version available for rollback")
    
    # Get previous version
    previous_version = versions[current_idx - 1]
    
    # Update current deployment
    deployment_info = {
        'deployment_dir': previous_version['deployment_path'],
        'model_path': str(Path(previous_version['deployment_path']) / "model.joblib"),
        'preprocessing_path': str(Path(previous_version['deployment_path']) / "preprocessing.joblib"),
        'metadata_path': str(Path(previous_version['deployment_path']) / "metadata.json"),
        'model_version': previous_version['version'],
        'deployed_at': datetime.now().isoformat()
    }
    
    # Save as current deployment
    deployment_info_path = MODELS_DIR / "current_deployment.json"
    with open(deployment_info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2, default=str)
    
    # Update registry
    registry['models'][model_name]['current_version'] = previous_version['version']
    registry['last_updated'] = datetime.now().isoformat()
    
    registry_file = MODELS_DIR / "registry" / "model_registry.json"
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    
    logger.info(f"Rolled back {model_name} to version {previous_version['version']}")
    
    return {
        'status': 'success',
        'previous_version': current_version,
        'current_version': previous_version['version'],
        'rollback_time': deployment_info['deployed_at']
    } 