"""
FastAPI Application for Komuniteti Predictive Maintenance Serving
"""

import time
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import click
from datetime import datetime

from ..schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    ModelMetrics
)
from ..pipelines.prediction_pipeline import make_single_prediction, make_batch_predictions
from ..steps.model_deployment import (
    load_current_deployment, 
    get_model_registry,
    rollback_to_previous_version
)
from ..config import api_config

# Initialize FastAPI app
app = FastAPI(
    title="Komuniteti Predictive Maintenance API",
    description="API for predicting building maintenance needs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching model
_model_cache = None
_model_metadata = None
_last_model_load = None

def load_model_if_needed():
    """Load model if not cached or if cache is stale"""
    global _model_cache, _model_metadata, _last_model_load
    
    current_time = time.time()
    
    # Load model if not cached or if cache is older than 5 minutes
    if (_model_cache is None or 
        _last_model_load is None or 
        current_time - _last_model_load > 300):
        
        try:
            model, artifacts = load_current_deployment()
            _model_cache = model
            _model_metadata = artifacts
            _last_model_load = current_time
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Model not available: {str(e)}"
            )
    
    return _model_cache, _model_metadata


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Komuniteti Predictive Maintenance API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        model, metadata = load_model_if_needed()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": metadata.get('metadata', {}).get('model_version', 'unknown'),
            "api_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_maintenance(request: PredictionRequest):
    """
    Predict maintenance needs for a single building asset.
    
    Args:
        request: Prediction request with building and asset information
        
    Returns:
        Prediction response with maintenance probability and risk assessment
    """
    
    try:
        # Ensure model is loaded
        load_model_if_needed()
        
        # Make prediction
        response = make_single_prediction(request)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_maintenance_batch(request: BatchPredictionRequest):
    """
    Predict maintenance needs for multiple building assets.
    
    Args:
        request: Batch prediction request with multiple buildings/assets
        
    Returns:
        Batch prediction response with all predictions
    """
    
    try:
        # Validate batch size
        if len(request.predictions) > api_config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.predictions)} exceeds maximum {api_config.max_batch_size}"
            )
        
        # Ensure model is loaded
        load_model_if_needed()
        
        # Record start time
        start_time = time.time()
        
        # Make predictions
        predictions = make_batch_predictions(request.predictions)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_seconds=processing_time
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the currently deployed model"""
    
    try:
        model, metadata = load_model_if_needed()
        
        model_info = {
            "model_name": metadata.get('metadata', {}).get('model_name', 'unknown'),
            "model_version": metadata.get('metadata', {}).get('model_version', 'unknown'),
            "model_type": metadata.get('metadata', {}).get('model_type', 'unknown'),
            "deployed_at": metadata.get('deployment_info', {}).get('deployed_at'),
            "feature_count": len(metadata.get('deployment_info', {}).get('feature_names', [])),
            "framework_versions": metadata.get('metadata', {}).get('framework_version', {}),
            "api_config": {
                "prediction_threshold": api_config.prediction_threshold,
                "low_risk_threshold": api_config.low_risk_threshold,
                "high_risk_threshold": api_config.high_risk_threshold
            }
        }
        
        return model_info
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model information not available: {str(e)}"
        )


@app.get("/model/metrics")
async def get_model_metrics():
    """Get performance metrics of the currently deployed model"""
    
    try:
        model, metadata = load_model_if_needed()
        
        # Extract metrics from metadata
        metrics_data = metadata.get('metadata', {}).get('metrics', {})
        
        if not metrics_data:
            raise HTTPException(
                status_code=404,
                detail="Model metrics not available"
            )
        
        return metrics_data
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model metrics not available: {str(e)}"
        )


@app.get("/model/registry")
async def get_model_registry():
    """Get model registry information"""
    
    try:
        registry = get_model_registry()
        return registry
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model registry: {str(e)}"
        )


@app.post("/model/rollback")
async def rollback_model(model_name: str, background_tasks: BackgroundTasks):
    """Rollback to the previous model version"""
    
    try:
        # Perform rollback
        rollback_result = rollback_to_previous_version(model_name)
        
        # Clear model cache to force reload
        global _model_cache, _model_metadata, _last_model_load
        _model_cache = None
        _model_metadata = None
        _last_model_load = None
        
        return {
            "message": "Model rollback successful",
            "rollback_info": rollback_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model rollback failed: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model():
    """Force reload of the model (useful after retraining)"""
    
    try:
        # Clear cache to force reload
        global _model_cache, _model_metadata, _last_model_load
        _model_cache = None
        _model_metadata = None
        _last_model_load = None
        
        # Load new model
        model, metadata = load_model_if_needed()
        
        return {
            "message": "Model reloaded successfully",
            "model_version": metadata.get('metadata', {}).get('model_version', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model reload failed: {str(e)}"
        )


# Laravel webhook endpoints

@app.post("/webhook/laravel/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Webhook endpoint for Laravel to trigger model retraining.
    This would typically be called when new data is available.
    """
    
    def run_retraining():
        """Background task to run retraining pipeline"""
        try:
            from ..pipelines.training_pipeline import komuniteti_training_pipeline
            
            # Run training pipeline with Laravel data
            pipeline_instance = komuniteti_training_pipeline(
                data_source="laravel_export",
                laravel_export=True,
                deploy_model=True
            )
            
            pipeline_instance.run()
            
        except Exception as e:
            # Log error (in production, you'd want proper logging)
            print(f"Retraining failed: {str(e)}")
    
    # Add retraining task to background
    background_tasks.add_task(run_retraining)
    
    return {
        "message": "Retraining triggered successfully",
        "status": "started",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/webhook/laravel/predictions/{building_id}")
async def get_building_predictions(building_id: int, asset_type: str = None):
    """
    Get all predictions for a specific building.
    This endpoint can be used by Laravel to fetch recent predictions.
    """
    
    # This is a placeholder - in a real implementation, you'd store
    # predictions in a database and query them here
    return {
        "building_id": building_id,
        "asset_type": asset_type,
        "message": "Prediction history endpoint - to be implemented with database storage",
        "timestamp": datetime.now().isoformat()
    }


# Error handlers

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Model not available",
            "detail": "No trained model found. Please train a model first.",
            "timestamp": datetime.now().isoformat()
        }
    )


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def main(host: str, port: int, reload: bool):
    """Run the Komuniteti Predictive Maintenance API server."""
    
    uvicorn.run(
        "src.api.serve:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main() 