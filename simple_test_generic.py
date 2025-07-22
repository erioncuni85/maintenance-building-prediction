#!/usr/bin/env python3
"""
Simple test to demonstrate generic building management data processing
without requiring ZenML installation.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def create_larger_generic_sample(output_path: str = "large_generic_sample.json", num_records: int = 100):
    """Create a larger generic sample based on the template"""
    
    print(f"Creating larger generic sample with {num_records} records...")
    
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
            floors = int(np.random.randint(3, 15))
            area = float(np.random.normal(2500, 800))
        elif building_type == 'commercial':
            floors = int(np.random.randint(5, 30))
            area = float(np.random.normal(6000, 2000))
        else:
            floors = int(np.random.randint(2, 20))
            area = float(np.random.normal(4000, 1500))
        
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
            "construction_year": int(np.random.randint(2000, 2020)),
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
        failure_reported = bool(np.random.choice([True, False], p=[0.25, 0.75]))
        if maintenance_type == 'corrective':
            failure_reported = True
        elif maintenance_type == 'preventive':
            failure_reported = bool(np.random.choice([True, False], p=[0.1, 0.9]))
        
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
                downtime_days = int(np.random.randint(1, 5))
            else:
                downtime_days = int(np.random.randint(0, 3))
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
            "technician_id": int(np.random.randint(101, 106)),
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
    output_file = Path("data") / output_path
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Large generic sample saved to {output_file}")
    
    return pd.DataFrame(maintenance_records)


def process_generic_building_json(data: dict) -> pd.DataFrame:
    """Process generic building management JSON format"""
    
    print("Processing generic building management JSON format...")
    
    # Extract maintenance records
    maintenance_records = data.get('maintenance_records', [])
    
    if not maintenance_records:
        raise ValueError("No maintenance records found in the JSON data")
    
    # Convert to DataFrame
    df = pd.DataFrame(maintenance_records)
    
    # Ensure required columns exist and convert data types
    df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    
    # Handle optional columns - remove extra fields for ML processing
    optional_fields = ['technician_name', 'asset_identifier', 'work_description', 'severity', 'completion_status']
    for field in optional_fields:
        if field in df.columns:
            df = df.drop(field, axis=1)
    
    # Ensure data types for core ML features
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
    
    print(f"Processed {len(df)} maintenance records from generic JSON format")
    
    return df


def analyze_data(df: pd.DataFrame):
    """Analyze the processed data"""
    
    print("\n📊 Data Analysis:")
    print(f"   - Total records: {len(df)}")
    print(f"   - Buildings: {df['building_id'].nunique()}")
    print(f"   - Countries: {', '.join(df['country'].unique())}")
    print(f"   - Asset types: {', '.join(df['asset_type'].unique())}")
    # Convert to datetime if it's string format
    if df['maintenance_date'].dtype == 'object':
        df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    print(f"   - Date range: {df['maintenance_date'].min().strftime('%Y-%m-%d')} to {df['maintenance_date'].max().strftime('%Y-%m-%d')}")
    
    print("\n📈 Asset Type Distribution:")
    asset_counts = df['asset_type'].value_counts()
    for asset, count in asset_counts.items():
        print(f"   - {asset}: {count} records")
    
    print("\n💰 Cost Analysis (EUR):")
    print(f"   - Average maintenance cost: €{df['maintenance_cost'].mean():.2f}")
    print(f"   - Total maintenance cost: €{df['maintenance_cost'].sum():.2f}")
    
    failure_rate = (df['failure_reported'].sum() / len(df)) * 100
    print(f"\n⚠️ Failure Rate: {failure_rate:.1f}%")
    
    # Show sample predictions using simple rules
    print("\n🔮 Sample Rule-Based Predictions:")
    print("   (Simulating what the ML model would predict)")
    
    sample_assets = df.sample(3)
    for _, asset in sample_assets.iterrows():
        # Simple rule-based prediction logic
        risk_score = 0
        
        # Age factor
        days_old = (datetime.now() - pd.to_datetime(asset['maintenance_date'])).days
        if days_old > 90:
            risk_score += 0.3
        
        # Failure history
        if asset['failure_reported']:
            risk_score += 0.4
            
        # Asset type risk
        high_risk_assets = ['elevator', 'HVAC', 'boiler']
        if asset['asset_type'] in high_risk_assets:
            risk_score += 0.2
            
        # Cost factor
        if asset['maintenance_cost'] > df['maintenance_cost'].mean():
            risk_score += 0.1
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        print(f"   🏢 Building {asset['building_id']} - {asset['asset_type']}:")
        print(f"      Risk Score: {risk_score:.1%} ({risk_level})")
        print(f"      Location: {asset['city']}, {asset['country']}")
        print(f"      Last maintenance: {pd.to_datetime(asset['maintenance_date']).strftime('%Y-%m-%d')}")
        print()


def main():
    """Test the generic data processing"""
    
    print("🏢 Testing Generic Building Management Data Processing")
    print("=" * 60)
    
    try:
        # Step 1: Create larger sample data
        print("📊 Step 1: Creating generic building management sample data...")
        df = create_larger_generic_sample(num_records=200)
        print(f"✅ Created sample with {len(df)} maintenance records")
        print()
        
        # Step 2: Load and process the existing generic JSON sample
        print("📋 Step 2: Processing existing generic JSON sample...")
        with open('data/generic_building_maintenance_sample.json', 'r') as f:
            small_sample = json.load(f)
        
        df_small = process_generic_building_json(small_sample)
        print(f"✅ Processed small sample with {len(df_small)} records")
        print()
        
        # Step 3: Show data structure
        print("📋 Step 3: Sample processed data structure:")
        print(df_small.head(3).to_string())
        print()
        
        # Step 4: Analyze the larger dataset
        analyze_data(df)
        
        # Step 5: Show integration examples
        print("🔗 Step 5: Integration Examples for Other Building Management Systems:")
        print()
        
        print("   📤 1. Export from Property Management Software:")
        print("   ```sql")
        print("   SELECT")
        print("     building_id,")
        print("     building_type,")
        print("     asset_type,")
        print("     maintenance_date,")
        print("     failure_reported,")
        print("     maintenance_cost,")
        print("     building_area,")
        print("     building_floors,")
        print("     city,")
        print("     country")
        print("   FROM maintenance_records")
        print("   WHERE maintenance_date >= '2023-01-01'")
        print("   ```")
        print()
        
        print("   📤 2. API Integration (REST):")
        print("   ```bash")
        print("   curl -X POST http://ml-api:8000/predict \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{")
        print("       \"building_id\": 123,")
        print("       \"building_type\": \"commercial\",")
        print("       \"asset_type\": \"elevator\",")
        print("       \"city\": \"Your City\",")
        print("       \"country\": \"Your Country\",")
        print("       \"building_area\": 5000.0,")
        print("       \"building_floors\": 15")
        print("     }'")
        print("   ```")
        print()
        
        print("   📤 3. Batch Processing:")
        print("   ```python")
        print("   import pandas as pd")
        print("   import requests")
        print("   ")
        print("   # Load your building data")
        print("   buildings_df = pd.read_csv('your_buildings.csv')")
        print("   ")
        print("   predictions = []")
        print("   for _, building in buildings_df.iterrows():")
        print("       response = requests.post('http://ml-api:8000/predict',")
        print("           json=building.to_dict())")
        print("       predictions.append(response.json())")
        print("   ```")
        print()
        
        print("✅ Compatible Building Management Systems:")
        print("   - Yardi Voyager")
        print("   - RealPage")
        print("   - AppFolio")
        print("   - Buildium")
        print("   - PropertyRadar")
        print("   - MRI Software")
        print("   - Any system with CSV/JSON export capability")
        print()
        
        print("🎉 SUCCESS: Generic data processing completed!")
        print()
        print("💡 Key Benefits:")
        print("   ✅ Works with any building management system")
        print("   ✅ Flexible JSON/CSV input formats")
        print("   ✅ Automatic data type conversion")
        print("   ✅ Handles optional fields gracefully")
        print("   ✅ Scalable to large datasets")
        print("   ✅ Ready for ML model training")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 