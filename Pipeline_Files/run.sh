#!/bin/bash
# Run script for testing cross-sell DAG without Airflow

# Ensure the script exits on error
set -e

# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "ERROR: Could not find activation script for virtual environment"
    exit 1
fi

# Install required dependencies
echo "Installing dependencies..."
pip install joblib pandas

# Create necessary directories if they don't exist
mkdir -p cross_sell_auto
mkdir -p data

# Check if required modules exist, create placeholder files if they don't
if [ ! -f "cross_sell_auto/__init__.py" ]; then
    echo "Creating placeholder files..."
    touch cross_sell_auto/__init__.py
    
    # Create basic config file if it doesn't exist
    if [ ! -f "cross_sell_auto/config.py" ]; then
        cat > cross_sell_auto/config.py <<EOF
"""
Configuration for Cross-sell Project
"""
class AIRFLOW_CONFIG:
    DAG_ID = "cross_sell_automobile_insurance" 
    START_DATE_DAYS_AGO = 1
    SCHEDULE_INTERVAL = "@daily"
    MAX_ACTIVE_RUNS = 1
    CATCHUP = False
    RETRIES = 3
    RETRY_DELAY_MINUTES = 5

class FILE_CONFIG:
    INPUT_CSV_PATH = "./data/input.csv"
    OUTPUT_PATH = "./data/output"

def validate_config():
    """Validate configuration settings"""
    return True
EOF
    fi
    
    # Create main runner placeholder
    if [ ! -f "cross_sell_auto/main_runner.py" ]; then
        cat > cross_sell_auto/main_runner.py <<EOF
"""
Main processor for cross-sell workflow
"""
class CrossSellProcessor:
    def __init__(self):
        pass
        
    def process_daily_file(self, target_date=None):
        """Process daily insurance data file"""
        # This is a mock implementation for testing
        return {
            "processing_date": target_date,
            "original_records": 1000,
            "new_clients": 150,
            "qualified_clients": 75,
            "leads_sent": 50,
            "successful_sends": 45,
            "errors": ["Sample error 1", "Sample error 2"],
            "summary": {
                "conversion_rate": 60.0,
                "top_cities": {
                    "New York": 10,
                    "Los Angeles": 8,
                    "Chicago": 7,
                    "Houston": 5,
                    "Phoenix": 4
                }
            }
        }
EOF
    fi
    
    # Create API client placeholder
    if [ ! -f "cross_sell_auto/telus_api_client.py" ]; then
        cat > cross_sell_auto/telus_api_client.py <<EOF
"""
Client for TelusHolding API
"""
class TelusAPIClient:
    def __init__(self):
        pass
        
    def test_connection(self):
        """Test API connection"""
        return True
EOF
    fi
    
    # Create test data file
    if [ ! -f "data/input.csv" ]; then
        cat > data/input.csv <<EOF
id,name,email,phone,postal_code,has_health_insurance,vehicle_age,annual_mileage,score
1,John Doe,john@example.com,555-1234,12345,true,3,12000,0.85
2,Jane Smith,jane@example.com,555-5678,23456,true,5,8000,0.72
3,Bob Johnson,bob@example.com,555-9012,34567,true,1,15000,0.91
EOF
    fi
fi

# Run the test
echo "Running test script..."
python test_cross_sell.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with errors."
    exit 1
fi 