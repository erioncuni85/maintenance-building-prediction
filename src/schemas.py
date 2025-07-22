"""
Data schemas for Komuniteti Predictive Maintenance Pipeline
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator
import pandas as pd

class MaintenanceRecord(BaseModel):
    """Schema for input maintenance records from Laravel"""
    
    building_id: int = Field(..., description="Building identifier")
    building_type: str = Field(..., description="Type of building (residential, commercial, etc.)")
    asset_type: str = Field(..., description="Type of asset (elevator, HVAC, boiler, etc.)")
    maintenance_type: str = Field(..., description="Type of maintenance (preventive, corrective)")
    maintenance_date: datetime = Field(..., description="Date of maintenance")
    failure_reported: bool = Field(..., description="Whether failure was reported")
    downtime_days: Optional[int] = Field(0, description="Days of downtime")
    maintenance_cost: Optional[float] = Field(0.0, description="Cost of maintenance")
    technician_id: Optional[int] = Field(None, description="Technician ID")
    period_days: Optional[int] = Field(None, description="Period between maintenance checks")
    city: str = Field(..., description="City where building is located")
    country: str = Field(..., description="Country where building is located")
    building_area: float = Field(..., description="Building area in square meters")
    building_floors: int = Field(..., description="Number of floors in building")
    
    @validator('building_area')
    def validate_building_area(cls, v):
        if v <= 0:
            raise ValueError('Building area must be positive')
        return v
    
    @validator('building_floors')
    def validate_building_floors(cls, v):
        if v <= 0:
            raise ValueError('Building floors must be positive')
        return v
    
    @validator('maintenance_cost')
    def validate_maintenance_cost(cls, v):
        if v < 0:
            raise ValueError('Maintenance cost cannot be negative')
        return v

class PredictionRequest(BaseModel):
    """Schema for prediction requests from Laravel"""
    
    building_id: int
    building_type: str
    asset_type: str
    city: str
    country: str
    building_area: float
    building_floors: int
    days_since_last_maintenance: Optional[int] = None
    maintenance_frequency: Optional[float] = None
    building_age_years: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "building_id": 123,
                "building_type": "residential",
                "asset_type": "elevator",
                "city": "Tirana",
                "country": "Albania",
                "building_area": 2500.0,
                "building_floors": 10,
                "days_since_last_maintenance": 45,
                "maintenance_frequency": 4.2,
                "building_age_years": 15.5
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction responses to Laravel"""
    
    building_id: int
    asset_type: str
    maintenance_probability: float = Field(..., ge=0, le=1, description="Probability of maintenance needed")
    predicted_timeframe_days: Optional[int] = Field(None, description="Predicted days until maintenance")
    risk_category: Literal["low", "medium", "high"] = Field(..., description="Risk categorization")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence")
    prediction_date: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(..., description="Version of the model used")
    
    class Config:
        schema_extra = {
            "example": {
                "building_id": 123,
                "asset_type": "elevator",
                "maintenance_probability": 0.75,
                "predicted_timeframe_days": 30,
                "risk_category": "high",
                "confidence_score": 0.89,
                "prediction_date": "2024-01-15T10:30:00",
                "model_version": "1.0.0"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests"""
    
    predictions: List[PredictionRequest] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "building_id": 123,
                        "building_type": "residential",
                        "asset_type": "elevator",
                        "city": "Tirana",
                        "country": "Albania",
                        "building_area": 2500.0,
                        "building_floors": 10
                    }
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses"""
    
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_seconds: float
    
class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    
    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_date: datetime
    training_samples: int
    feature_importance: dict
    
class DataValidationReport(BaseModel):
    """Schema for data validation reports"""
    
    total_records: int
    valid_records: int
    invalid_records: int
    missing_data_percentage: float
    validation_errors: List[str]
    data_quality_score: float = Field(..., ge=0, le=1)
    
def validate_dataframe_schema(df: pd.DataFrame) -> DataValidationReport:
    """Validate a pandas DataFrame against the expected schema"""
    
    total_records = len(df)
    validation_errors = []
    
    # Check required columns
    required_columns = [
        'building_id', 'building_type', 'asset_type', 'maintenance_type',
        'maintenance_date', 'failure_reported', 'building_area', 'building_floors'
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        validation_errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'building_id' in df.columns and not pd.api.types.is_integer_dtype(df['building_id']):
        validation_errors.append("building_id must be integer type")
    
    if 'building_area' in df.columns and not pd.api.types.is_numeric_dtype(df['building_area']):
        validation_errors.append("building_area must be numeric type")
    
    if 'building_floors' in df.columns and not pd.api.types.is_integer_dtype(df['building_floors']):
        validation_errors.append("building_floors must be integer type")
    
    # Check for negative values
    if 'building_area' in df.columns and (df['building_area'] <= 0).any():
        validation_errors.append("building_area contains non-positive values")
    
    if 'building_floors' in df.columns and (df['building_floors'] <= 0).any():
        validation_errors.append("building_floors contains non-positive values")
    
    # Calculate missing data percentage
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    # Calculate valid records (no missing values in required columns)
    valid_mask = df[required_columns].notna().all(axis=1) if all(col in df.columns for col in required_columns) else df.notna().all(axis=1)
    valid_records = valid_mask.sum()
    invalid_records = total_records - valid_records
    
    # Calculate data quality score
    quality_score = max(0, 1 - (len(validation_errors) * 0.1 + missing_percentage * 0.01))
    
    return DataValidationReport(
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        missing_data_percentage=missing_percentage,
        validation_errors=validation_errors,
        data_quality_score=quality_score
    ) 