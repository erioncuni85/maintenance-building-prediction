"""
Data Preprocessing Step for Komuniteti Predictive Maintenance Pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, Annotated, Dict, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

from zenml import step
from zenml.logger import get_logger

from ..config import model_config, data_config, MODELS_DIR

logger = get_logger(__name__)

@step
def data_preprocessing_step(
    validated_data: pd.DataFrame,
    data_quality_metrics: Dict
) -> Tuple[
    Annotated[pd.DataFrame, "processed_features"],
    Annotated[pd.Series, "target_variable"],
    Annotated[Dict[str, Any], "preprocessing_artifacts"],
    Annotated[Dict, "feature_info"]
]:
    """
    Preprocess and engineer features for predictive maintenance model.
    
    Args:
        validated_data: Cleaned dataframe from validation step
        data_quality_metrics: Quality metrics from validation
        
    Returns:
        Tuple of processed features, target variable, preprocessing artifacts, and feature info
    """
    
    logger.info("Starting data preprocessing and feature engineering")
    
    df = validated_data.copy()
    preprocessing_artifacts = {}
    
    try:
        # 1. Create target variable
        df, target = _create_target_variable(df)
        logger.info(f"Created target variable with {target.sum()} positive samples out of {len(target)}")
        
        # 2. Feature engineering
        df = _engineer_time_features(df)
        df = _engineer_maintenance_features(df)
        df = _engineer_building_features(df)
        df = _engineer_aggregation_features(df)
        
        # 3. Encode categorical variables
        df, categorical_encoders = _encode_categorical_features(df)
        preprocessing_artifacts['categorical_encoders'] = categorical_encoders
        
        # 4. Scale numerical features
        df, numerical_scaler = _scale_numerical_features(df)
        preprocessing_artifacts['numerical_scaler'] = numerical_scaler
        
        # 5. Feature selection and final preparation
        df, feature_info = _prepare_final_features(df)
        
        # Save preprocessing artifacts
        _save_preprocessing_artifacts(preprocessing_artifacts)
        
        logger.info(f"Preprocessing completed. Final feature matrix shape: {df.shape}")
        logger.info(f"Features created: {list(df.columns)}")
        
        return df, target, preprocessing_artifacts, feature_info
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


def _create_target_variable(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create target variable for maintenance prediction"""
    
    logger.info("Creating target variable")
    
    # Sort by building_id, asset_type, and maintenance_date
    df = df.sort_values(['building_id', 'asset_type', 'maintenance_date'])
    
    # Calculate days until next maintenance for each building-asset combination
    def calculate_next_maintenance_days(group):
        group = group.sort_values('maintenance_date')
        group['days_to_next_maintenance'] = group['maintenance_date'].diff().shift(-1).dt.days
        return group
    
    df = df.groupby(['building_id', 'asset_type']).apply(calculate_next_maintenance_days)
    df = df.reset_index(drop=True)
    
    # Create binary target: maintenance needed within prediction horizon
    prediction_horizon_days = model_config.prediction_horizon_months * 30
    
    # Target is True if:
    # 1. Next maintenance is within prediction horizon, OR
    # 2. There was a failure reported, OR
    # 3. It's been longer than expected since last maintenance
    target = (
        (df['days_to_next_maintenance'] <= prediction_horizon_days) |
        (df['failure_reported'] == True) |
        (df['period_days'].notna() & 
         (df.groupby(['building_id', 'asset_type'])['maintenance_date'].diff().dt.days > df['period_days'] * 1.2))
    ).fillna(False)
    
    # Remove the helper column
    df = df.drop('days_to_next_maintenance', axis=1)
    
    return df, target


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-based features"""
    
    logger.info("Engineering time-based features")
    
    if not data_config.create_time_features:
        return df
    
    # Calculate days since last maintenance for each building-asset combination
    df = df.sort_values(['building_id', 'asset_type', 'maintenance_date'])
    
    df['days_since_last_maintenance'] = df.groupby(['building_id', 'asset_type'])['maintenance_date'].diff().dt.days
    df['days_since_last_maintenance'] = df['days_since_last_maintenance'].fillna(df['period_days'].fillna(90))
    
    # Calculate maintenance frequency (maintenances per year)
    def calculate_maintenance_frequency(group):
        if len(group) < 2:
            return pd.Series([2.0] * len(group), index=group.index)  # Default frequency
        
        date_range = (group['maintenance_date'].max() - group['maintenance_date'].min()).days
        if date_range < 30:  # Less than a month of data
            return pd.Series([2.0] * len(group), index=group.index)
        
        frequency = len(group) / (date_range / 365.25)
        return pd.Series([frequency] * len(group), index=group.index)
    
    df['maintenance_frequency'] = df.groupby(['building_id', 'asset_type']).apply(calculate_maintenance_frequency)
    df['maintenance_frequency'] = df['maintenance_frequency'].reset_index(level=[0, 1], drop=True)
    
    # Time-based features from maintenance_date
    df['maintenance_month'] = df['maintenance_date'].dt.month
    df['maintenance_quarter'] = df['maintenance_date'].dt.quarter
    df['maintenance_day_of_week'] = df['maintenance_date'].dt.dayofweek
    df['maintenance_season'] = df['maintenance_month'].apply(_get_season)
    
    # Calculate building age at time of maintenance
    # Note: This requires construction_date - using a placeholder for now
    current_date = datetime.now()
    df['building_age_years'] = (df['maintenance_date'] - pd.to_datetime('2010-01-01')).dt.days / 365.25
    
    return df


def _engineer_maintenance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer maintenance-specific features"""
    
    logger.info("Engineering maintenance-specific features")
    
    # Maintenance cost per square meter
    df['maintenance_cost_per_sqm'] = df['maintenance_cost'] / df['building_area']
    df['maintenance_cost_per_sqm'] = df['maintenance_cost_per_sqm'].fillna(0)
    
    # Maintenance cost categories
    df['maintenance_cost_category'] = pd.cut(
        df['maintenance_cost'], 
        bins=[0, 200, 500, 1000, np.inf], 
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Downtime ratio (downtime per maintenance cost)
    df['downtime_cost_ratio'] = df['downtime_days'] / (df['maintenance_cost'] + 1)  # +1 to avoid division by zero
    
    # Failure rate features
    def calculate_failure_rate(group):
        failure_rate = group['failure_reported'].mean()
        return pd.Series([failure_rate] * len(group), index=group.index)
    
    df['building_failure_rate'] = df.groupby('building_id').apply(calculate_failure_rate)
    df['building_failure_rate'] = df['building_failure_rate'].reset_index(level=0, drop=True)
    
    df['asset_failure_rate'] = df.groupby('asset_type').apply(calculate_failure_rate)
    df['asset_failure_rate'] = df['asset_failure_rate'].reset_index(level=0, drop=True)
    
    # Maintenance type effectiveness (inverse of failure rate for preventive maintenance)
    preventive_mask = df['maintenance_type'] == 'preventive'
    df['preventive_effectiveness'] = 0.0
    if preventive_mask.any():
        preventive_failure_rates = df[preventive_mask].groupby(['building_id', 'asset_type'])['failure_reported'].transform('mean')
        df.loc[preventive_mask, 'preventive_effectiveness'] = 1 - preventive_failure_rates
    
    return df


def _engineer_building_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer building-specific features"""
    
    logger.info("Engineering building-specific features")
    
    # Building complexity metrics
    df['area_per_floor'] = df['building_area'] / df['building_floors']
    
    # Building size categories
    df['building_size_category'] = pd.cut(
        df['building_area'],
        bins=[0, 1000, 3000, 6000, np.inf],
        labels=['small', 'medium', 'large', 'very_large']
    )
    
    # Building height categories
    df['building_height_category'] = pd.cut(
        df['building_floors'],
        bins=[0, 3, 7, 15, np.inf],
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Asset density (number of different asset types per building)
    asset_density = df.groupby('building_id')['asset_type'].nunique().to_dict()
    df['asset_density'] = df['building_id'].map(asset_density)
    
    # Location-based features
    # City size proxy (number of buildings per city)
    city_building_count = df.groupby('city')['building_id'].nunique().to_dict()
    df['city_building_count'] = df['city'].map(city_building_count)
    
    return df


def _engineer_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer aggregation features"""
    
    logger.info("Engineering aggregation features")
    
    if not data_config.create_aggregation_features:
        return df
    
    # Rolling statistics for each building-asset combination
    df = df.sort_values(['building_id', 'asset_type', 'maintenance_date'])
    
    # Rolling averages of maintenance cost (last 3 maintenances)
    df['maintenance_cost_rolling_mean'] = df.groupby(['building_id', 'asset_type'])['maintenance_cost'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Rolling averages of downtime (last 3 maintenances)
    df['downtime_rolling_mean'] = df.groupby(['building_id', 'asset_type'])['downtime_days'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Trend indicators (is maintenance cost/downtime increasing?)
    df['cost_trend'] = df.groupby(['building_id', 'asset_type'])['maintenance_cost'].transform(
        lambda x: x.diff().fillna(0)
    )
    
    df['downtime_trend'] = df.groupby(['building_id', 'asset_type'])['downtime_days'].transform(
        lambda x: x.diff().fillna(0)
    )
    
    # Cumulative maintenance metrics
    df['cumulative_maintenance_cost'] = df.groupby(['building_id', 'asset_type'])['maintenance_cost'].cumsum()
    df['cumulative_downtime'] = df.groupby(['building_id', 'asset_type'])['downtime_days'].cumsum()
    df['maintenance_count'] = df.groupby(['building_id', 'asset_type']).cumcount() + 1
    
    return df


def _encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Encode categorical features"""
    
    logger.info("Encoding categorical features")
    
    categorical_encoders = {}
    categorical_features = model_config.categorical_features
    
    # Filter to only existing columns
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    for col in categorical_features:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            # Use Label Encoding for ordinal features and high-cardinality features
            if col in ['technician_id', 'building_id'] or df[col].nunique() > 10:
                encoder = LabelEncoder()
                df[col + '_encoded'] = encoder.fit_transform(df[col].astype(str))
                categorical_encoders[col] = encoder
                
            else:
                # Use One-Hot Encoding for nominal features with low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(df[[col]])
                
                # Create column names
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                
                # Add encoded features to dataframe
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                df = pd.concat([df, encoded_df], axis=1)
                
                categorical_encoders[col] = encoder
    
    # Encode remaining categorical columns that were created during feature engineering
    new_categorical_cols = ['maintenance_cost_category', 'building_size_category', 
                           'building_height_category', 'maintenance_season']
    
    for col in new_categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            encoder = LabelEncoder()
            df[col + '_encoded'] = encoder.fit_transform(df[col].astype(str))
            categorical_encoders[col] = encoder
    
    return df, categorical_encoders


def _scale_numerical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features"""
    
    logger.info("Scaling numerical features")
    
    # Identify numerical columns to scale
    numerical_cols = []
    
    # Original numerical features
    base_numerical = [col for col in model_config.numeric_features if col in df.columns]
    numerical_cols.extend(base_numerical)
    
    # Engineered numerical features
    engineered_numerical = [
        'days_since_last_maintenance', 'maintenance_frequency', 'building_age_years',
        'maintenance_cost_per_sqm', 'downtime_cost_ratio', 'building_failure_rate',
        'asset_failure_rate', 'preventive_effectiveness', 'area_per_floor',
        'asset_density', 'city_building_count', 'maintenance_cost_rolling_mean',
        'downtime_rolling_mean', 'cost_trend', 'downtime_trend',
        'cumulative_maintenance_cost', 'cumulative_downtime', 'maintenance_count'
    ]
    
    for col in engineered_numerical:
        if col in df.columns:
            numerical_cols.append(col)
    
    # Remove duplicates
    numerical_cols = list(set(numerical_cols))
    
    # Scale the features
    scaler = StandardScaler()
    
    if numerical_cols:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
    else:
        logger.warning("No numerical features found to scale")
    
    return df, scaler


def _prepare_final_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Prepare final feature matrix"""
    
    logger.info("Preparing final feature matrix")
    
    # Define columns to exclude from the final feature matrix
    exclude_cols = [
        'building_id', 'maintenance_date', 'failure_reported',  # IDs and target-related
        'building_type', 'asset_type', 'maintenance_type', 'city', 'country',  # Original categorical (now encoded)
        'maintenance_cost_category', 'building_size_category', 'building_height_category', 'maintenance_season'  # Encoded versions exist
    ]
    
    # Keep only feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure all features are numeric
    final_features = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            final_features.append(col)
        else:
            logger.warning(f"Excluding non-numeric column: {col}")
    
    df_features = df[final_features].copy()
    
    # Handle any remaining NaN values
    df_features = df_features.fillna(0)
    
    # Create feature info dictionary
    feature_info = {
        'feature_names': list(df_features.columns),
        'n_features': len(df_features.columns),
        'feature_types': {
            col: str(df_features[col].dtype) for col in df_features.columns
        },
        'feature_statistics': {
            col: {
                'mean': float(df_features[col].mean()),
                'std': float(df_features[col].std()),
                'min': float(df_features[col].min()),
                'max': float(df_features[col].max())
            } for col in df_features.columns
        }
    }
    
    logger.info(f"Final feature matrix shape: {df_features.shape}")
    
    return df_features, feature_info


def _save_preprocessing_artifacts(artifacts: Dict[str, Any]) -> None:
    """Save preprocessing artifacts for later use"""
    
    logger.info("Saving preprocessing artifacts")
    
    # Save each artifact
    for name, artifact in artifacts.items():
        artifact_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(artifact, artifact_path)
        logger.info(f"Saved {name} to {artifact_path}")


def _get_season(month: int) -> str:
    """Get season from month number"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'


# Utility function for loading preprocessing artifacts
def load_preprocessing_artifacts() -> Dict[str, Any]:
    """Load saved preprocessing artifacts"""
    
    artifacts = {}
    
    # Load categorical encoders
    encoders_path = MODELS_DIR / "categorical_encoders.joblib"
    if encoders_path.exists():
        artifacts['categorical_encoders'] = joblib.load(encoders_path)
    
    # Load numerical scaler
    scaler_path = MODELS_DIR / "numerical_scaler.joblib"
    if scaler_path.exists():
        artifacts['numerical_scaler'] = joblib.load(scaler_path)
    
    return artifacts


def preprocess_new_data(
    new_data: pd.DataFrame,
    preprocessing_artifacts: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Preprocess new data using saved artifacts for prediction.
    
    Args:
        new_data: New data to preprocess
        preprocessing_artifacts: Saved preprocessing artifacts
        
    Returns:
        Preprocessed feature matrix
    """
    
    if preprocessing_artifacts is None:
        preprocessing_artifacts = load_preprocessing_artifacts()
    
    df = new_data.copy()
    
    # Apply the same feature engineering steps (without target creation)
    df = _engineer_time_features(df)
    df = _engineer_maintenance_features(df)
    df = _engineer_building_features(df)
    df = _engineer_aggregation_features(df)
    
    # Apply saved encoders
    if 'categorical_encoders' in preprocessing_artifacts:
        encoders = preprocessing_artifacts['categorical_encoders']
        
        for col, encoder in encoders.items():
            if col in df.columns:
                if isinstance(encoder, LabelEncoder):
                    # Handle unknown categories in LabelEncoder
                    df[col + '_encoded'] = df[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
                else:  # OneHotEncoder
                    encoded_features = encoder.transform(df[[col]])
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
    
    # Apply saved scaler
    if 'numerical_scaler' in preprocessing_artifacts:
        scaler = preprocessing_artifacts['numerical_scaler']
        numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Only scale columns that were in the original training set
        scale_cols = [col for col in numerical_cols if col in scaler.feature_names_in_]
        if scale_cols:
            df[scale_cols] = scaler.transform(df[scale_cols])
    
    # Prepare final features (same logic as training)
    df_features, _ = _prepare_final_features(df)
    
    return df_features 