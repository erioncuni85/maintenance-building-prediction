"""
Data Ingestion Step for Komuniteti Predictive Maintenance Pipeline
"""

import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Union, Annotated
import logging

from zenml import step
from zenml.logger import get_logger

from ..config import DATA_DIR, DATABASE_CONFIG, LARAVEL_CONFIG
from ..schemas import validate_dataframe_schema, DataValidationReport

logger = get_logger(__name__)

@step
def data_ingestion_step(
    data_source: str = "csv",
    file_path: str = None,
    laravel_export: bool = False
) -> Tuple[
    Annotated[pd.DataFrame, "raw_data"],
    Annotated[DataValidationReport, "validation_report"]
]:
    """
    Ingest maintenance data from various sources.
    
    Args:
        data_source: Type of data source ('csv', 'json', 'database')
        file_path: Path to the data file (relative to DATA_DIR)
        laravel_export: Whether to fetch data from Laravel API
        
    Returns:
        Tuple of raw dataframe and validation report
    """
    
    logger.info(f"Starting data ingestion from {data_source}")
    
    df = None
    
    try:
        if laravel_export:
            df = _fetch_from_laravel()
        elif data_source == "csv":
            df = _load_csv_data(file_path)
        elif data_source == "json":
            df = _load_json_data(file_path)
        elif data_source == "database":
            df = _load_from_database()
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
            
        logger.info(f"Successfully loaded {len(df)} records from {data_source}")
        
        # Validate the loaded data
        validation_report = validate_dataframe_schema(df)
        logger.info(f"Data validation completed. Quality score: {validation_report.data_quality_score:.2f}")
        
        if validation_report.validation_errors:
            logger.warning(f"Data validation issues found: {validation_report.validation_errors}")
        
        return df, validation_report
        
    except Exception as e:
        logger.error(f"Failed to ingest data from {data_source}: {str(e)}")
        raise


def _load_csv_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    
    if file_path is None:
        file_path = "maintenance_data.csv"
    
    full_path = DATA_DIR / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"CSV file not found: {full_path}")
    
    logger.info(f"Loading CSV data from {full_path}")
    
    # Load CSV with proper data types
    df = pd.read_csv(
        full_path,
        parse_dates=['maintenance_date'],
        dtype={
            'building_id': 'int64',
            'building_type': 'string',
            'asset_type': 'string', 
            'maintenance_type': 'string',
            'failure_reported': 'bool',
            'downtime_days': 'Int64',  # Nullable integer
            'maintenance_cost': 'float64',
            'technician_id': 'Int64',
            'period_days': 'Int64',
            'city': 'string',
            'country': 'string',
            'building_area': 'float64',
            'building_floors': 'int64'
        }
    )
    
    return df


def _load_json_data(file_path: str) -> pd.DataFrame:
    """Load data from JSON file"""
    
    if file_path is None:
        file_path = "maintenance_data.json"
    
    full_path = DATA_DIR / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"JSON file not found: {full_path}")
    
    logger.info(f"Loading JSON data from {full_path}")
    
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's the generic building management format
    if isinstance(data, dict) and 'maintenance_records' in data:
        df = _process_generic_building_json(data)
    else:
        # Convert to DataFrame (legacy format)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame([data])
    
        # Convert date columns
        if 'maintenance_date' in df.columns:
            df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    
    return df


def _process_generic_building_json(data: dict) -> pd.DataFrame:
    """Process generic building management JSON format"""
    
    logger.info("Processing generic building management JSON format")
    
    # Extract maintenance records
    maintenance_records = data.get('maintenance_records', [])
    
    if not maintenance_records:
        raise ValueError("No maintenance records found in the JSON data")
    
    # Convert to DataFrame
    df = pd.DataFrame(maintenance_records)
    
    # Ensure required columns exist and convert data types
    df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    
    # Handle optional columns
    if 'technician_name' in df.columns:
        df = df.drop('technician_name', axis=1)  # Remove name, keep ID
    
    if 'asset_identifier' in df.columns:
        df = df.drop('asset_identifier', axis=1)  # Optional field
    
    if 'work_description' in df.columns:
        df = df.drop('work_description', axis=1)  # Optional field
        
    if 'severity' in df.columns:
        df = df.drop('severity', axis=1)  # Optional field
        
    if 'completion_status' in df.columns:
        df = df.drop('completion_status', axis=1)  # Optional field
    
    # Ensure data types
    type_mapping = {
        'building_id': 'int64',
        'building_type': 'string',
        'asset_type': 'string',
        'maintenance_type': 'string',
        'failure_reported': 'bool',
        'downtime_days': 'Int64',
        'maintenance_cost': 'float64',
        'technician_id': 'Int64',
        'period_days': 'Int64',
        'city': 'string',
        'country': 'string',
        'building_area': 'float64',
        'building_floors': 'int64'
    }
    
    for col, dtype in type_mapping.items():
        if col in df.columns:
            if dtype == 'Int64':  # Nullable integer
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif dtype == 'bool':
                df[col] = df[col].astype('bool')
            elif dtype == 'string':
                df[col] = df[col].astype('string')
            elif dtype in ['int64', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    logger.info(f"Processed {len(df)} maintenance records from generic JSON format")
    
    return df


def _load_from_database() -> pd.DataFrame:
    """Load data directly from Laravel MySQL database"""
    
    try:
        import pymysql
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError("pymysql and sqlalchemy required for database connection")
    
    logger.info("Loading data from MySQL database")
    
    # Create database connection
    connection_string = (
        f"mysql+pymysql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
    )
    
    engine = create_engine(connection_string)
    
    # SQL query to extract maintenance data from Laravel database
    query = """
    SELECT 
        b.id as building_id,
        bt.name as building_type,
        s.title as asset_type,
        'preventive' as maintenance_type,
        rs.checked_date as maintenance_date,
        CASE WHEN r.id IS NOT NULL THEN 1 ELSE 0 END as failure_reported,
        0 as downtime_days,
        COALESCE(ri.value, 0) as maintenance_cost,
        bst.user_id as technician_id,
        bst.period_days,
        c.name as city,
        co.name as country,
        b.area as building_area,
        b.number_of_floors as building_floors
    FROM buildings b
    LEFT JOIN building_types bt ON b.type = bt.code
    LEFT JOIN building_service_technician bst ON b.id = bst.building_id
    LEFT JOIN services s ON bst.service_id = s.id
    LEFT JOIN recurring_services rs ON s.id = rs.service_id AND b.id = rs.building_id
    LEFT JOIN reports r ON b.id = r.building_id AND r.created_at >= rs.checked_date - INTERVAL 30 DAY
    LEFT JOIN resident_invoice ri ON ri.resident_id IN (
        SELECT resident_id FROM building_resident WHERE building_id = b.id LIMIT 1
    )
    LEFT JOIN cities c ON b.city_id = c.id
    LEFT JOIN countries co ON b.country_id = co.id
    WHERE rs.checked_date IS NOT NULL
      AND b.deleted_at IS NULL
      AND s.deleted_at IS NULL
    ORDER BY b.id, rs.checked_date
    """
    
    df = pd.read_sql(query, engine)
    
    # Clean up the data
    df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    df['failure_reported'] = df['failure_reported'].astype(bool)
    
    logger.info(f"Loaded {len(df)} records from database")
    
    return df


def _fetch_from_laravel() -> pd.DataFrame:
    """Fetch data from Laravel API export endpoint"""
    
    try:
        import requests
    except ImportError:
        raise ImportError("requests library required for Laravel API integration")
    
    logger.info("Fetching data from Laravel API")
    
    url = LARAVEL_CONFIG['api_base_url'] + LARAVEL_CONFIG['export_endpoint']
    headers = {
        'Authorization': f"Bearer {LARAVEL_CONFIG['api_token']}",
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Handle Laravel API response format
    if 'data' in data:
        df = pd.DataFrame(data['data'])
    else:
        df = pd.DataFrame(data)
    
    # Convert date columns
    if 'maintenance_date' in df.columns:
        df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    
    logger.info(f"Fetched {len(df)} records from Laravel API")
    
    return df


# Additional utility function for creating sample data
def create_sample_data(output_path: str = "sample_maintenance_data.csv", num_records: int = 1000):
    """Create sample maintenance data for testing"""
    
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.info(f"Creating sample data with {num_records} records")
    
    np.random.seed(42)
    
    # Sample data generation
    building_types = ['residential', 'commercial', 'mixed', 'industrial']
    asset_types = ['elevator', 'HVAC', 'boiler', 'plumbing', 'electrical', 'fire_system']
    maintenance_types = ['preventive', 'corrective', 'emergency']
    cities = ['Tirana', 'Durres', 'Vlore', 'Shkoder', 'Fier']
    countries = ['Albania']
    
    data = []
    start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
    
    for i in range(num_records):
        # Generate correlated data
        building_id = np.random.randint(1, 101)  # 100 different buildings
        building_type = np.random.choice(building_types)
        asset_type = np.random.choice(asset_types)
        
        # Building characteristics
        if building_type == 'residential':
            building_floors = np.random.randint(3, 15)
            building_area = np.random.normal(2000, 500)
        elif building_type == 'commercial':
            building_floors = np.random.randint(5, 25)
            building_area = np.random.normal(5000, 1500)
        else:
            building_floors = np.random.randint(2, 20)
            building_area = np.random.normal(3000, 1000)
        
        building_area = max(500, building_area)  # Minimum area
        
        # Maintenance data
        maintenance_date = start_date + timedelta(days=np.random.randint(0, 730))
        failure_reported = np.random.choice([True, False], p=[0.3, 0.7])
        maintenance_type = np.random.choice(maintenance_types, p=[0.6, 0.3, 0.1])
        
        # Correlated costs and downtime (in EUR)
        if asset_type == 'elevator':
            base_cost = 800
            max_downtime = 5
        elif asset_type == 'HVAC':
            base_cost = 600
            max_downtime = 3
        else:
            base_cost = 400
            max_downtime = 2
        
        maintenance_cost = np.random.exponential(base_cost)
        downtime_days = np.random.randint(0, max_downtime + 1) if failure_reported else 0
        
        record = {
            'building_id': building_id,
            'building_type': building_type,
            'asset_type': asset_type,
            'maintenance_type': maintenance_type,
            'maintenance_date': maintenance_date,
            'failure_reported': failure_reported,
            'downtime_days': downtime_days,
            'maintenance_cost': round(maintenance_cost, 2),
            'technician_id': np.random.randint(1, 21),
            'period_days': np.random.choice([30, 60, 90, 120]) if maintenance_type == 'preventive' else None,
            'city': np.random.choice(cities),
            'country': np.random.choice(countries),
            'building_area': round(building_area, 2),
            'building_floors': building_floors
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = DATA_DIR / output_path
    df.to_csv(output_file, index=False)
    
    logger.info(f"Sample data saved to {output_file}")
    
    return df


def create_larger_generic_sample(output_path: str = "large_generic_sample.json", num_records: int = 200):
    """Create a larger generic sample based on the template"""
    
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.info(f"Creating larger generic sample with {num_records} records")
    
    np.random.seed(42)
    
    # Base template
    building_types = ['residential', 'commercial', 'mixed', 'industrial']
    asset_types = ['elevator', 'HVAC', 'boiler', 'plumbing', 'electrical', 'fire_system', 'security_system', 'lighting']
    maintenance_types = ['preventive', 'corrective', 'emergency']
    cities = ['Metropolitan City', 'Riverside', 'Downtown', 'Uptown', 'Suburbia']
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Albania']
    
    buildings = []
    maintenance_records = []
    start_date = datetime.now() - timedelta(days=365*2)
    
    # Generate buildings
    for i in range(1, 21):  # 20 buildings
        building_type = np.random.choice(building_types)
        city = np.random.choice(cities)
        country = np.random.choice(countries)
        
        if building_type == 'residential':
            floors = np.random.randint(3, 15)
            area = np.random.normal(2500, 800)
        elif building_type == 'commercial':
            floors = np.random.randint(5, 30)
            area = np.random.normal(6000, 2000)
        else:
            floors = np.random.randint(2, 20)
            area = np.random.normal(4000, 1500)
        
        area = max(1000, area)
        
        building = {
            "building_id": i,
            "building_name": f"{building_type.title()} Building {i}",
            "building_type": building_type,
            "address": f"{100 + i} {city} Street",
            "city": city,
            "country": country,
            "postal_code": f"{10000 + i}",
            "building_area": round(area, 1),
            "building_floors": floors,
            "construction_year": np.random.randint(2000, 2020),
            "total_units": floors * 10
        }
        buildings.append(building)
    
    # Generate maintenance records
    for i in range(num_records):
        building = np.random.choice(buildings)
        asset_type = np.random.choice(asset_types)
        maintenance_type = np.random.choice(maintenance_types, p=[0.7, 0.25, 0.05])
        
        # Generate maintenance date
        maintenance_date = start_date + timedelta(days=np.random.randint(0, 730))
        
        # Failure correlation
        failure_reported = np.random.choice([True, False], p=[0.25, 0.75])
        if maintenance_type == 'corrective':
            failure_reported = True
        elif maintenance_type == 'preventive':
            failure_reported = np.random.choice([True, False], p=[0.1, 0.9])
        
        # Cost correlation (in EUR)
        base_costs = {
            'elevator': 900, 'HVAC': 700, 'boiler': 800, 'plumbing': 350,
            'electrical': 600, 'fire_system': 500, 'security_system': 300, 'lighting': 250
        }
        base_cost = base_costs.get(asset_type, 500)
        
        if maintenance_type == 'corrective':
            cost_multiplier = np.random.uniform(1.5, 3.0)
        elif maintenance_type == 'emergency':
            cost_multiplier = np.random.uniform(2.0, 4.0)
        else:
            cost_multiplier = np.random.uniform(0.8, 1.2)
        
        maintenance_cost = base_cost * cost_multiplier
        
        # Downtime correlation
        if failure_reported:
            if asset_type in ['elevator', 'HVAC', 'boiler']:
                downtime_days = np.random.randint(1, 5)
            else:
                downtime_days = np.random.randint(0, 3)
        else:
            downtime_days = 0
        
        # Period days for preventive maintenance
        period_days = None
        if maintenance_type == 'preventive':
            period_mapping = {
                'elevator': 90, 'HVAC': 90, 'boiler': 120, 'plumbing': 180,
                'electrical': 120, 'fire_system': 120, 'security_system': 180, 'lighting': 365
            }
            period_days = period_mapping.get(asset_type, 120)
        
        record = {
            "record_id": i + 1,
            "building_id": building["building_id"],
            "building_type": building["building_type"],
            "asset_type": asset_type,
            "asset_identifier": f"{asset_type.upper()}-{building['building_id']:03d}",
            "maintenance_type": maintenance_type,
            "maintenance_date": maintenance_date.isoformat(),
            "failure_reported": failure_reported,
            "downtime_days": downtime_days,
            "maintenance_cost": round(maintenance_cost, 2),
            "technician_id": np.random.randint(101, 106),
            "technician_name": np.random.choice(["John Smith", "Maria Garcia", "David Chen", "Robert Johnson", "Sarah Wilson"]),
            "period_days": period_days,
            "city": building["city"],
            "country": building["country"],
            "building_area": building["building_area"],
            "building_floors": building["building_floors"],
            "work_description": f"{maintenance_type.title()} maintenance for {asset_type}",
            "severity": np.random.choice(["low", "medium", "high"], p=[0.5, 0.35, 0.15]),
            "completion_status": "completed"
        }
        
        maintenance_records.append(record)
    
    # Create the complete JSON structure
    data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "system_name": "Generic Building Management System - Extended Sample",
            "version": "1.0",
            "description": "Extended sample maintenance data for predictive maintenance ML training",
            "data_period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d")
            },
            "total_records": len(maintenance_records)
        },
        "buildings": buildings,
        "maintenance_records": maintenance_records
    }
    
    # Save to file
    output_file = DATA_DIR / output_path
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Large generic sample saved to {output_file}")
    
    return pd.DataFrame(maintenance_records) 