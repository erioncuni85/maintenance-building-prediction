"""
Training Pipeline for Komuniteti Predictive Maintenance
"""

import click
from zenml import pipeline
from zenml.logger import get_logger

from ..steps import (
    data_ingestion_step,
    data_validation_step,
    data_preprocessing_step,
    model_training_step,
    model_evaluation_step,
    model_deployment_step
)
from ..config import zenml_config

logger = get_logger(__name__)

@pipeline(name=zenml_config.pipeline_name)
def komuniteti_training_pipeline(
    data_source: str = "csv",
    file_path: str = None,
    laravel_export: bool = False,
    deploy_model: bool = True
):
    """
    Complete training pipeline for predictive maintenance model.
    
    Args:
        data_source: Type of data source ('csv', 'json', 'database', 'laravel')
        file_path: Path to data file (if using file-based source)
        laravel_export: Whether to fetch data from Laravel API
        deploy_model: Whether to deploy the trained model
    """
    
    logger.info("Starting Komuniteti predictive maintenance training pipeline")
    
    # Step 1: Data Ingestion
    raw_data, validation_report = data_ingestion_step(
        data_source=data_source,
        file_path=file_path,
        laravel_export=laravel_export
    )
    
    # Step 2: Data Validation
    validated_data, data_quality_metrics = data_validation_step(
        raw_data=raw_data,
        validation_report=validation_report
    )
    
    # Step 3: Data Preprocessing
    processed_features, target_variable, preprocessing_artifacts, feature_info = data_preprocessing_step(
        validated_data=validated_data,
        data_quality_metrics=data_quality_metrics
    )
    
    # Step 4: Model Training
    trained_models, best_model_metrics, best_model_name = model_training_step(
        processed_features=processed_features,
        target_variable=target_variable,
        preprocessing_artifacts=preprocessing_artifacts,
        feature_info=feature_info
    )
    
    # Step 5: Model Evaluation
    evaluation_report, model_comparison, evaluation_plots_path = model_evaluation_step(
        trained_models=trained_models,
        best_model_metrics=best_model_metrics,
        best_model_name=best_model_name,
        processed_features=processed_features,
        target_variable=target_variable
    )
    
    # Step 6: Model Deployment (optional)
    if deploy_model:
        deployment_status, deployment_info, model_version = model_deployment_step(
            trained_models=trained_models,
            best_model_metrics=best_model_metrics,
            best_model_name=best_model_name,
            preprocessing_artifacts=preprocessing_artifacts,
            feature_info=feature_info
        )
        
        logger.info(f"Pipeline completed. Model {best_model_name} v{model_version} status: {deployment_status}")
    else:
        logger.info(f"Pipeline completed. Best model: {best_model_name} (not deployed)")


@click.command()
@click.option(
    "--data-source", 
    default="csv",
    type=click.Choice(["csv", "json", "database"]),
    help="Data source type"
)
@click.option(
    "--file-path",
    default=None,
    help="Path to data file (relative to data directory)"
)
@click.option(
    "--laravel-export",
    is_flag=True,
    help="Fetch data from Laravel API"
)
@click.option(
    "--no-deploy",
    is_flag=True,
    help="Skip model deployment"
)
@click.option(
    "--create-sample-data",
    is_flag=True,
    help="Create sample data for testing"
)
def main(data_source: str, file_path: str, laravel_export: bool, no_deploy: bool, create_sample_data: bool):
    """Run the Komuniteti predictive maintenance training pipeline."""
    
    # Create sample data if requested
    if create_sample_data:
        from ..steps.data_ingestion import create_sample_data
        create_sample_data()
        logger.info("Sample data created successfully")
        return
    
    # Run the pipeline
    try:
        pipeline_instance = komuniteti_training_pipeline(
            data_source=data_source,
            file_path=file_path,
            laravel_export=laravel_export,
            deploy_model=not no_deploy
        )
        
        pipeline_instance.run()
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 