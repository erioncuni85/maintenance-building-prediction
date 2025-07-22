"""
Prediction Pipeline for Komuniteti Predictive Maintenance
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import click
from zenml import pipeline, step
from zenml.logger import get_logger

from ..steps.data_preprocessing import preprocess_new_data
from ..steps.model_deployment import load_current_deployment
from ..steps.model_training import predict_maintenance_probability, categorize_risk
from ..config import api_config
from ..schemas import PredictionRequest, PredictionResponse

logger = get_logger(__name__)

@step
def prediction_step(
    prediction_requests: List[Dict[str, Any]]
) -> List[PredictionResponse]:
    """
    Make predictions for maintenance needs.
    
    Args:
        prediction_requests: List of prediction request dictionaries
        
    Returns:
        List of prediction responses
    """
    
    logger.info(f"Making predictions for {len(prediction_requests)} requests")
    
    try:
        # Load the deployed model
        model, deployment_artifacts = load_current_deployment()
        preprocessing = deployment_artifacts['preprocessing']
        metadata = deployment_artifacts['metadata']
        
        # Convert requests to DataFrame
        df = pd.DataFrame(prediction_requests)
        
        # Preprocess the data
        processed_features = preprocess_new_data(df, preprocessing)
        
        # Make predictions
        probabilities = predict_maintenance_probability(model, processed_features, return_probabilities=True)
        predictions = predict_maintenance_probability(model, processed_features, return_probabilities=False)
        
        # Create responses
        responses = []
        for i, request in enumerate(prediction_requests):
            probability = float(probabilities[i])
            risk_category = categorize_risk(
                probability, 
                api_config.low_risk_threshold, 
                api_config.high_risk_threshold
            )
            
            # Estimate timeframe (simplified logic)
            if probability > 0.7:
                predicted_timeframe_days = 30
            elif probability > 0.5:
                predicted_timeframe_days = 60
            elif probability > 0.3:
                predicted_timeframe_days = 90
            else:
                predicted_timeframe_days = None
            
            response = PredictionResponse(
                building_id=request['building_id'],
                asset_type=request['asset_type'],
                maintenance_probability=probability,
                predicted_timeframe_days=predicted_timeframe_days,
                risk_category=risk_category,
                confidence_score=min(0.95, max(0.6, probability * 1.2)),  # Simplified confidence
                model_version=metadata['model_version']
            )
            
            responses.append(response)
        
        logger.info(f"Generated {len(responses)} predictions successfully")
        
        return responses
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


@pipeline
def komuniteti_prediction_pipeline(
    prediction_requests: List[Dict[str, Any]]
) -> List[PredictionResponse]:
    """
    Prediction pipeline for making maintenance predictions.
    
    Args:
        prediction_requests: List of prediction request dictionaries
        
    Returns:
        List of prediction responses
    """
    
    logger.info("Starting Komuniteti prediction pipeline")
    
    predictions = prediction_step(prediction_requests=prediction_requests)
    
    return predictions


def make_single_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Make a single prediction (convenience function).
    
    Args:
        request: Single prediction request
        
    Returns:
        Single prediction response
    """
    
    # Convert to dictionary format
    request_dict = request.dict()
    
    # Run prediction pipeline
    pipeline_instance = komuniteti_prediction_pipeline([request_dict])
    results = pipeline_instance.run()
    
    return results[0]


def make_batch_predictions(requests: List[PredictionRequest]) -> List[PredictionResponse]:
    """
    Make batch predictions.
    
    Args:
        requests: List of prediction requests
        
    Returns:
        List of prediction responses
    """
    
    # Convert to dictionary format
    request_dicts = [req.dict() for req in requests]
    
    # Run prediction pipeline
    pipeline_instance = komuniteti_prediction_pipeline(request_dicts)
    results = pipeline_instance.run()
    
    return results


@click.command()
@click.option(
    "--building-id",
    required=True,
    type=int,
    help="Building ID"
)
@click.option(
    "--asset-type",
    required=True,
    help="Asset type (elevator, HVAC, etc.)"
)
@click.option(
    "--building-type",
    required=True,
    help="Building type"
)
@click.option(
    "--city",
    required=True,
    help="City name"
)
@click.option(
    "--country",
            default="Albania",
    help="Country name"
)
@click.option(
    "--building-area",
    required=True,
    type=float,
    help="Building area in square meters"
)
@click.option(
    "--building-floors",
    required=True,
    type=int,
    help="Number of building floors"
)
@click.option(
    "--days-since-last-maintenance",
    type=int,
    default=None,
    help="Days since last maintenance"
)
@click.option(
    "--maintenance-frequency",
    type=float,
    default=None,
    help="Maintenance frequency per year"
)
@click.option(
    "--building-age-years",
    type=float,
    default=None,
    help="Building age in years"
)
def main(
    building_id: int,
    asset_type: str,
    building_type: str,
    city: str,
    country: str,
    building_area: float,
    building_floors: int,
    days_since_last_maintenance: int,
    maintenance_frequency: float,
    building_age_years: float
):
    """Run a single prediction using the command line."""
    
    try:
        # Create prediction request
        request = PredictionRequest(
            building_id=building_id,
            building_type=building_type,
            asset_type=asset_type,
            city=city,
            country=country,
            building_area=building_area,
            building_floors=building_floors,
            days_since_last_maintenance=days_since_last_maintenance,
            maintenance_frequency=maintenance_frequency,
            building_age_years=building_age_years
        )
        
        # Make prediction
        response = make_single_prediction(request)
        
        # Display results
        print("\n=== Maintenance Prediction Results ===")
        print(f"Building ID: {response.building_id}")
        print(f"Asset Type: {response.asset_type}")
        print(f"Maintenance Probability: {response.maintenance_probability:.2%}")
        print(f"Risk Category: {response.risk_category.upper()}")
        print(f"Predicted Timeframe: {response.predicted_timeframe_days} days" if response.predicted_timeframe_days else "No immediate maintenance needed")
        print(f"Confidence Score: {response.confidence_score:.2%}")
        print(f"Model Version: {response.model_version}")
        print(f"Prediction Date: {response.prediction_date}")
        print("=====================================\n")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 