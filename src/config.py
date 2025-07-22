"""
Configuration file for Komuniteti Predictive Maintenance Pipeline
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """Model training configuration"""
    
    # Target variable
    target_column: str = "maintenance_needed"
    prediction_horizon_months: int = 3
    
    # Features to use
    numeric_features: List[str] = None
    categorical_features: List[str] = None
    
    # Model parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Model algorithms to try
    models_to_train: List[str] = None
    
    def __post_init__(self):
        if self.numeric_features is None:
            self.numeric_features = [
                "building_area",
                "building_floors", 
                "maintenance_cost",
                "downtime_days",
                "period_days",
                "days_since_last_maintenance",
                "maintenance_frequency",
                "building_age_years"
            ]
            
        if self.categorical_features is None:
            self.categorical_features = [
                "building_type",
                "asset_type", 
                "maintenance_type",
                "city",
                "country",
                "technician_id"
            ]
            
        if self.models_to_train is None:
            self.models_to_train = [
                "random_forest",
                "xgboost", 
                "lightgbm"
            ]

@dataclass
class DataConfig:
    """Data processing configuration"""
    
    # Input data schema
    required_columns: List[str] = None
    
    # Data validation
    max_missing_percentage: float = 0.3
    min_records_per_building: int = 10
    
    # Feature engineering
    create_time_features: bool = True
    create_aggregation_features: bool = True
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                "building_id",
                "building_type", 
                "asset_type",
                "maintenance_type",
                "maintenance_date",
                "failure_reported",
                "building_area",
                "building_floors"
            ]

@dataclass
class APIConfig:
    """API serving configuration"""
    
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "komuniteti_maintenance_predictor"
    prediction_threshold: float = 0.5
    
    # Risk categorization thresholds
    low_risk_threshold: float = 0.3
    high_risk_threshold: float = 0.7

@dataclass
class ZenMLConfig:
    """ZenML pipeline configuration"""
    
    pipeline_name: str = "komuniteti_maintenance_pipeline"
    experiment_tracker: str = "mlflow"
    model_registry: str = "mlflow"
    
    # Stack components (can be overridden)
    orchestrator: str = "local"
    artifact_store: str = "local"

# Environment-based configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_DATABASE", "komuniteti"),
    "username": os.getenv("DB_USERNAME", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Laravel integration
LARAVEL_CONFIG = {
    "api_base_url": os.getenv("LARAVEL_API_URL", "http://localhost:8000/api"),
    "api_token": os.getenv("LARAVEL_API_TOKEN", ""),
    "export_endpoint": "/maintenance/export",
    "prediction_webhook": "/maintenance/predictions",
}

# Global configuration instances
model_config = ModelConfig()
data_config = DataConfig()
api_config = APIConfig()
zenml_config = ZenMLConfig() 