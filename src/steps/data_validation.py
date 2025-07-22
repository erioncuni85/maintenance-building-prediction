"""
Data Validation Step for Komuniteti Predictive Maintenance Pipeline
"""

import pandas as pd
import numpy as np
from typing import Tuple, Annotated, Dict, List
from datetime import datetime, timedelta

from zenml import step
from zenml.logger import get_logger

from ..config import data_config
from ..schemas import DataValidationReport

logger = get_logger(__name__)

@step
def data_validation_step(
    raw_data: pd.DataFrame,
    validation_report: DataValidationReport
) -> Tuple[
    Annotated[pd.DataFrame, "validated_data"],
    Annotated[Dict, "data_quality_metrics"]
]:
    """
    Validate and clean the raw maintenance data.
    
    Args:
        raw_data: Raw dataframe from data ingestion
        validation_report: Initial validation report
        
    Returns:
        Tuple of validated dataframe and quality metrics
    """
    
    logger.info("Starting data validation and cleaning")
    
    # Start with a copy of the raw data
    df = raw_data.copy()
    initial_records = len(df)
    
    # Track data quality metrics
    quality_metrics = {
        'initial_records': initial_records,
        'validation_steps': [],
        'records_removed': {},
        'data_quality_issues': [],
        'final_data_quality_score': 0.0
    }
    
    try:
        # 1. Check for required columns
        df, step_metrics = _validate_required_columns(df)
        quality_metrics['validation_steps'].append('required_columns')
        quality_metrics['records_removed']['missing_columns'] = step_metrics['removed']
        
        # 2. Validate data types
        df, step_metrics = _validate_data_types(df)
        quality_metrics['validation_steps'].append('data_types')
        quality_metrics['records_removed']['invalid_types'] = step_metrics['removed']
        
        # 3. Handle missing values
        df, step_metrics = _handle_missing_values(df)
        quality_metrics['validation_steps'].append('missing_values')
        quality_metrics['records_removed']['excessive_missing'] = step_metrics['removed']
        
        # 4. Validate business logic
        df, step_metrics = _validate_business_logic(df)
        quality_metrics['validation_steps'].append('business_logic')
        quality_metrics['records_removed']['business_logic'] = step_metrics['removed']
        
        # 5. Remove duplicates
        df, step_metrics = _remove_duplicates(df)
        quality_metrics['validation_steps'].append('duplicates')
        quality_metrics['records_removed']['duplicates'] = step_metrics['removed']
        
        # 6. Validate date ranges
        df, step_metrics = _validate_date_ranges(df)
        quality_metrics['validation_steps'].append('date_ranges')
        quality_metrics['records_removed']['invalid_dates'] = step_metrics['removed']
        
        # 7. Check minimum records per building
        df, step_metrics = _validate_minimum_records_per_building(df)
        quality_metrics['validation_steps'].append('minimum_records')
        quality_metrics['records_removed']['insufficient_building_data'] = step_metrics['removed']
        
        # Calculate final metrics
        final_records = len(df)
        total_removed = initial_records - final_records
        retention_rate = final_records / initial_records if initial_records > 0 else 0
        
        quality_metrics.update({
            'final_records': final_records,
            'total_removed': total_removed,
            'retention_rate': retention_rate,
            'final_data_quality_score': _calculate_data_quality_score(df, quality_metrics)
        })
        
        logger.info(f"Data validation completed. Retained {final_records}/{initial_records} records ({retention_rate:.2%})")
        logger.info(f"Final data quality score: {quality_metrics['final_data_quality_score']:.2f}")
        
        return df, quality_metrics
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise


def _validate_required_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate presence of required columns"""
    
    logger.info("Validating required columns")
    
    required_cols = data_config.required_columns
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        # For missing columns, we'll fill with default values where possible
        for col in missing_cols:
            if col in ['downtime_days', 'maintenance_cost']:
                df[col] = 0
            elif col in ['technician_id', 'period_days']:
                df[col] = None
            else:
                logger.error(f"Cannot proceed without required column: {col}")
                raise ValueError(f"Required column missing: {col}")
    
    # Remove rows where essential columns are still missing
    essential_cols = ['building_id', 'building_type', 'asset_type', 'maintenance_date']
    initial_len = len(df)
    df = df.dropna(subset=essential_cols)
    removed = initial_len - len(df)
    
    logger.info(f"Required columns validation: {removed} records removed")
    
    return df, {'removed': removed}


def _validate_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate and convert data types"""
    
    logger.info("Validating data types")
    
    initial_len = len(df)
    
    # Convert data types with error handling
    try:
        # Numeric columns
        numeric_cols = ['building_area', 'building_floors', 'maintenance_cost', 'downtime_days']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Integer columns
        integer_cols = ['building_id', 'building_floors', 'downtime_days', 'technician_id', 'period_days']
        for col in integer_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # String columns
        string_cols = ['building_type', 'asset_type', 'maintenance_type', 'city', 'country']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('string')
        
        # Date columns
        if 'maintenance_date' in df.columns:
            df['maintenance_date'] = pd.to_datetime(df['maintenance_date'], errors='coerce')
        
        # Boolean columns
        if 'failure_reported' in df.columns:
            df['failure_reported'] = df['failure_reported'].astype('bool')
    
    except Exception as e:
        logger.warning(f"Data type conversion issues: {str(e)}")
    
    # Remove rows with invalid essential data types
    before_len = len(df)
    df = df.dropna(subset=['building_id', 'maintenance_date'])
    removed = before_len - len(df)
    
    logger.info(f"Data type validation: {removed} records removed")
    
    return df, {'removed': removed}


def _handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Handle missing values based on business logic"""
    
    logger.info("Handling missing values")
    
    initial_len = len(df)
    
    # Calculate missing percentage per row
    missing_percentage = df.isnull().sum(axis=1) / len(df.columns)
    
    # Remove rows with excessive missing data
    max_missing = data_config.max_missing_percentage
    before_len = len(df)
    df = df[missing_percentage <= max_missing]
    removed_excessive = before_len - len(df)
    
    # Fill missing values with business logic
    # Fill downtime_days: 0 for non-failure events
    if 'downtime_days' in df.columns and 'failure_reported' in df.columns:
        df.loc[df['failure_reported'] == False, 'downtime_days'] = df.loc[df['failure_reported'] == False, 'downtime_days'].fillna(0)
    
    # Fill maintenance_cost: median by asset_type
    if 'maintenance_cost' in df.columns:
        df['maintenance_cost'] = df.groupby('asset_type')['maintenance_cost'].transform(
            lambda x: x.fillna(x.median())
        )
        # If still missing, fill with overall median
        df['maintenance_cost'] = df['maintenance_cost'].fillna(df['maintenance_cost'].median())
    
    # Fill period_days: mode by asset_type for preventive maintenance
    if 'period_days' in df.columns and 'maintenance_type' in df.columns:
        preventive_mask = df['maintenance_type'] == 'preventive'
        df.loc[preventive_mask, 'period_days'] = df.loc[preventive_mask].groupby('asset_type')['period_days'].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 90)
        )
    
    # Fill technician_id with most frequent for the building
    if 'technician_id' in df.columns:
        df['technician_id'] = df.groupby('building_id')['technician_id'].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 1)
        )
    
    logger.info(f"Missing values handling: {removed_excessive} records removed for excessive missing data")
    
    return df, {'removed': removed_excessive}


def _validate_business_logic(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate business logic constraints"""
    
    logger.info("Validating business logic")
    
    initial_len = len(df)
    
    # Remove records with invalid values
    invalid_mask = pd.Series(False, index=df.index)
    
    # Building area must be positive
    if 'building_area' in df.columns:
        invalid_mask |= (df['building_area'] <= 0)
    
    # Building floors must be positive
    if 'building_floors' in df.columns:
        invalid_mask |= (df['building_floors'] <= 0)
    
    # Maintenance cost cannot be negative
    if 'maintenance_cost' in df.columns:
        invalid_mask |= (df['maintenance_cost'] < 0)
    
    # Downtime days cannot be negative
    if 'downtime_days' in df.columns:
        invalid_mask |= (df['downtime_days'] < 0)
    
    # If failure_reported is False, downtime_days should be 0 or very low
    if 'failure_reported' in df.columns and 'downtime_days' in df.columns:
        invalid_mask |= ((df['failure_reported'] == False) & (df['downtime_days'] > 1))
    
    # Period days should be reasonable (between 7 and 365 days)
    if 'period_days' in df.columns:
        invalid_mask |= ((df['period_days'] < 7) | (df['period_days'] > 365))
    
    # Remove invalid records
    df = df[~invalid_mask]
    removed = initial_len - len(df)
    
    logger.info(f"Business logic validation: {removed} records removed")
    
    return df, {'removed': removed}


def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Remove duplicate records"""
    
    logger.info("Removing duplicates")
    
    initial_len = len(df)
    
    # Define columns for duplicate detection
    duplicate_cols = ['building_id', 'asset_type', 'maintenance_date', 'maintenance_type']
    duplicate_cols = [col for col in duplicate_cols if col in df.columns]
    
    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=duplicate_cols, keep='first')
    
    removed = initial_len - len(df)
    
    logger.info(f"Duplicate removal: {removed} records removed")
    
    return df, {'removed': removed}


def _validate_date_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate date ranges"""
    
    logger.info("Validating date ranges")
    
    initial_len = len(df)
    
    if 'maintenance_date' in df.columns:
        # Remove future dates (more than 1 day in the future to account for time zones)
        future_threshold = datetime.now() + timedelta(days=1)
        future_mask = df['maintenance_date'] > future_threshold
        
        # Remove very old dates (more than 10 years ago)
        old_threshold = datetime.now() - timedelta(days=10*365)
        old_mask = df['maintenance_date'] < old_threshold
        
        # Remove invalid dates
        invalid_dates_mask = future_mask | old_mask
        df = df[~invalid_dates_mask]
    
    removed = initial_len - len(df)
    
    logger.info(f"Date range validation: {removed} records removed")
    
    return df, {'removed': removed}


def _validate_minimum_records_per_building(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ensure minimum number of records per building"""
    
    logger.info("Validating minimum records per building")
    
    initial_len = len(df)
    min_records = data_config.min_records_per_building
    
    if 'building_id' in df.columns:
        # Count records per building
        building_counts = df['building_id'].value_counts()
        
        # Identify buildings with insufficient data
        insufficient_buildings = building_counts[building_counts < min_records].index
        
        # Remove records from buildings with insufficient data
        df = df[~df['building_id'].isin(insufficient_buildings)]
    
    removed = initial_len - len(df)
    
    logger.info(f"Minimum records validation: {removed} records removed from buildings with insufficient data")
    
    return df, {'removed': removed}


def _calculate_data_quality_score(df: pd.DataFrame, quality_metrics: Dict) -> float:
    """Calculate overall data quality score"""
    
    # Base score from retention rate
    retention_score = quality_metrics['retention_rate']
    
    # Penalty for missing values
    missing_penalty = 0
    if len(df) > 0:
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        missing_penalty = missing_percentage * 0.5
    
    # Bonus for good data distribution
    distribution_bonus = 0
    if len(df) > 100:  # Only if we have sufficient data
        # Check if we have good distribution across key dimensions
        if 'building_id' in df.columns:
            building_variety = df['building_id'].nunique()
            if building_variety >= 10:
                distribution_bonus += 0.1
        
        if 'asset_type' in df.columns:
            asset_variety = df['asset_type'].nunique()
            if asset_variety >= 3:
                distribution_bonus += 0.1
    
    # Calculate final score
    quality_score = max(0, min(1, retention_score - missing_penalty + distribution_bonus))
    
    return quality_score 