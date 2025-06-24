#!/bin/bash
# Test script for cross-sell DAG without Airflow (Mac version)

echo "===== STARTING CROSS-SELL TEST ====="

# Ensure the script exits if a command fails
set -e

# Show actual commands being executed
set -x

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required dependencies
echo "Installing dependencies..."
pip install numpy pandas joblib scikit-learn

# Prepare directory structure
echo "Setting up directory structure..."
mkdir -p cross_sell_auto
mkdir -p data
mkdir -p models
mkdir -p reports

# Generate the dummy model if it doesn't exist
echo "Generating dummy model..."
python models/dummy_model.py

# Set environment variables
export TELUS_API_KEY="test_api_key_123"
export TELUS_USE_TEST="true"

# Run the test
echo "Running test..."
python test_cross_sell.py

echo "===== TEST COMPLETED =====" 