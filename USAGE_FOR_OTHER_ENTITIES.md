# Usage Guide for Other Building Management Systems

This guide is specifically for **building management companies and property managers** who want to implement predictive maintenance using this ZenML pipeline, regardless of their current software system.

## 🏢 Who Can Use This

- **Property Management Companies** (Yardi Voyager, RealPage, AppFolio, Buildium users)
- **Facility Management Firms**
- **Real Estate Investment Trusts (REITs)**
- **Commercial Building Operators**
- **Residential Property Managers**
- **Any organization managing building maintenance**

## 📊 Quick Start for Non-Komuniteti Systems

### Step 1: Export Your Data

Your building management system needs to export data in this format. Most systems can export to CSV or JSON.

**Required Fields:**
```json
{
  "building_id": 123,
  "building_type": "commercial",
  "asset_type": "elevator", 
  "maintenance_type": "preventive",
  "maintenance_date": "2023-12-15T08:00:00Z",
  "failure_reported": false,
  "downtime_days": 0,
  "maintenance_cost": 650.00,
  "technician_id": 102,
  "period_days": 90,
  "city": "Your City",
  "country": "Your Country",
  "building_area": 4500.0,
  "building_floors": 12
}
```

### Step 2: Use the Sample Structure

Use the provided sample files as templates:

1. **Small Sample:** `data/generic_building_maintenance_sample.json`
   - Perfect for testing and understanding the structure
   - Contains 13 sample records with buildings and maintenance data

2. **Large Sample:** `data/large_generic_sample.json` 
   - Generated with 200+ records for comprehensive testing
   - Shows variety across different countries, assets, and costs

### Step 3: Test with Your Data

```bash
# 1. Clone this repository
git clone https://github.com/erioncuni85/komuniteti-maintenance-prediction.git
cd komuniteti-maintenance-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test with the provided samples
python3 simple_test_generic.py

# 4. Replace sample data with your exported data
# Copy your JSON file to data/your_building_data.json

# 5. Run the pipeline with your data
python3 -c "
from src.steps.data_ingestion import data_ingestion_step
df, report = data_ingestion_step(
    data_source='json', 
    file_path='data/your_building_data.json'
)
print(f'Processed {len(df)} records')
print(report)
"
```

## 🔌 Integration Examples by System

### For Yardi Voyager Users
```sql
-- Export query for Yardi Voyager
SELECT 
    p.hProp as building_id,
    pt.sPropType as building_type,
    wot.sDesc as asset_type,
    CASE wo.iType 
        WHEN 1 THEN 'preventive'
        WHEN 2 THEN 'corrective'
        ELSE 'emergency'
    END as maintenance_type,
    wo.dtCompleted as maintenance_date,
    CASE WHEN wo.iUrgency > 3 THEN 1 ELSE 0 END as failure_reported,
    DATEDIFF(day, wo.dtStart, wo.dtCompleted) as downtime_days,
    wo.deCost as maintenance_cost,
    wo.hEmployee as technician_id,
    p.sAddr1 as address,
    p.sCity as city,
    p.sCountry as country,
    p.dArea as building_area,
    p.iFloors as building_floors
FROM WorkOrder wo
JOIN Property p ON wo.hProp = p.hProp
JOIN PropertyType pt ON p.hPropType = pt.hPropType
JOIN WorkOrderType wot ON wo.hWOType = wot.hWOType
WHERE wo.dtCompleted >= '2023-01-01'
```

### For RealPage Users
```sql
-- Export query for RealPage
SELECT 
    pm.PropertyID as building_id,
    p.PropertyType as building_type,
    wo.CategoryDesc as asset_type,
    wo.WorkType as maintenance_type,
    wo.CompletedDate as maintenance_date,
    CASE WHEN wo.Priority = 'Emergency' THEN 1 ELSE 0 END as failure_reported,
    wo.DowntimeDays as downtime_days,
    wo.TotalCost as maintenance_cost,
    wo.TechnicianID as technician_id,
    p.City as city,
    p.Country as country,
    p.TotalSqFt as building_area,
    p.Stories as building_floors
FROM WorkOrders wo
JOIN PropertyMaintenance pm ON wo.PropertyMaintenanceID = pm.ID
JOIN Properties p ON pm.PropertyID = p.PropertyID
WHERE wo.CompletedDate >= '2023-01-01'
```

### For AppFolio Users
```python
# Python script for AppFolio API integration
import requests
import json
from datetime import datetime

def export_appfolio_data(api_token, portfolio_id):
    headers = {'Authorization': f'Bearer {api_token}'}
    
    # Get maintenance requests
    maintenance_url = f"https://api.appfolio.com/v1/portfolios/{portfolio_id}/maintenance_requests"
    response = requests.get(maintenance_url, headers=headers)
    
    maintenance_data = []
    for request in response.json()['maintenance_requests']:
        record = {
            "building_id": request['property']['id'],
            "building_type": request['property']['property_type'],
            "asset_type": request['category'],
            "maintenance_type": request['priority'].lower(),
            "maintenance_date": request['completed_at'],
            "failure_reported": request['priority'] == 'Emergency',
            "downtime_days": calculate_downtime(request),
            "maintenance_cost": float(request['total_cost'] or 0),
            "technician_id": request['assigned_to']['id'] if request['assigned_to'] else None,
            "city": request['property']['address']['city'],
            "country": request['property']['address']['country'] or 'USA',
            "building_area": float(request['property']['square_feet'] or 0),
            "building_floors": int(request['property']['stories'] or 1)
        }
        maintenance_data.append(record)
    
    return maintenance_data
```

### For Buildium Users
```python
# Buildium API integration example
import requests

def export_buildium_data(api_key, portfolio_id):
    headers = {
        'X-Buildium-Api-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    # Get work orders
    url = "https://api.buildium.com/v1/workorders"
    params = {
        'propertyids': portfolio_id,
        'statuses': 'Completed',
        'fromdate': '2023-01-01'
    }
    
    response = requests.get(url, headers=headers, params=params)
    work_orders = response.json()
    
    # Transform to our format
    maintenance_data = []
    for wo in work_orders:
        record = {
            "building_id": wo['Property']['Id'],
            "building_type": wo['Property']['PropertyType'],
            "asset_type": wo['Category'],
            "maintenance_type": "preventive" if wo['Type'] == 'Preventive' else "corrective",
            "maintenance_date": wo['CompletedDateTime'],
            "failure_reported": wo['Priority'] == 'Emergency',
            "maintenance_cost": wo['TotalCost'],
            "city": wo['Property']['Address']['City'],
            "country": wo['Property']['Address']['Country'],
            "building_area": wo['Property']['RentableArea'],
            "building_floors": wo['Property']['FloorCount']
        }
        maintenance_data.append(record)
    
    return maintenance_data
```

## 🌍 Currency and Localization

### Euro (EUR) Support
The system now supports EUR currency by default:
- All costs are displayed in Euros (€)
- Suitable for European building management companies
- Includes Albania and other European countries

### Adding Your Country/Currency
1. Update the countries list in `src/steps/data_ingestion.py`:
```python
countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Albania', 'YourCountry']
```

2. Modify cost display in analysis functions to show your currency symbol.

## 📈 Running Predictions

### Option 1: Quick Test with Sample Data
```bash
python3 simple_test_generic.py
```

### Option 2: Full Pipeline with Your Data
```bash
# 1. Install ZenML
pip install zenml[server]==0.55.5

# 2. Initialize ZenML
zenml init

# 3. Run training pipeline
python3 -m src.pipelines.training_pipeline --data-source json --file-path data/your_data.json

# 4. Start prediction API
python3 -m src.api.serve
```

### Option 3: Docker Deployment
```bash
# Build and run with Docker
docker build -t building-maintenance-ml .
docker run -p 8000:8000 building-maintenance-ml
```

## 🔗 API Integration

Once deployed, integrate with your existing system:

```python
import requests

# Single building prediction
response = requests.post('http://your-api:8000/predict', json={
    "building_id": 123,
    "building_type": "commercial",
    "asset_type": "elevator",
    "city": "Your City",
    "country": "Your Country",
    "building_area": 5000.0,
    "building_floors": 15
})

prediction = response.json()
print(f"Maintenance probability: {prediction['maintenance_probability']}")
print(f"Risk category: {prediction['risk_category']}")
```

## 📞 Support

- **Technical Issues:** Open an issue on GitHub
- **Integration Help:** Check the `README.md` for detailed setup
- **Deployment Questions:** See `DEPLOYMENT.md` for production guidelines

## ✅ Success Stories

This pipeline structure is compatible with:
- ✅ Multi-tenant property management systems
- ✅ International building portfolios
- ✅ Mixed-use property types (residential, commercial, industrial)
- ✅ Various asset types (HVAC, elevators, plumbing, electrical, etc.)
- ✅ Different maintenance strategies (preventive, corrective, emergency)

**Start with the sample data, adapt to your format, and deploy for immediate predictive maintenance insights!** 🚀 