#!/usr/bin/env python
"""
Simple test script for Cross-sell workflow without Airflow
This script manually executes each function from the DAG
"""
import os
import sys
import json
from datetime import datetime

print("===== STARTING SIMPLE CROSS-SELL TEST =====")

# Set up environment variables directly
os.environ['TELUS_API_KEY'] = 'test_api_key_for_simple_test'
os.environ['TELUS_USE_TEST'] = 'true'

# Create a simple xcom mock
class SimpleMockTaskInstance:
    def __init__(self):
        self.data = {}
    
    def xcom_push(self, key, value):
        print(f"Storing data with key: {key}")
        self.data[key] = value
    
    def xcom_pull(self, task_ids=None, key=None):
        print(f"Retrieving data with key: {key}")
        return self.data.get(key)

# Print directory info
print(f"Current working directory: {os.getcwd()}")
print("Files in directory:")
for item in os.listdir():
    print(f"  {item}")

# Check if cross_sell_dag.py exists
if not os.path.exists("cross_sell_dag.py"):
    print("ERROR: cross_sell_dag.py not found!")
    sys.exit(1)

# Ensure cross_sell_auto directory is in path
if os.path.exists("cross_sell_auto"):
    sys.path.append("./cross_sell_auto")
    print("Added cross_sell_auto to Python path")
else:
    print("Creating cross_sell_auto directory")
    os.makedirs("cross_sell_auto", exist_ok=True)
    
    # Create necessary files
    if not os.path.exists("cross_sell_auto/__init__.py"):
        with open("cross_sell_auto/__init__.py", "w") as f:
            f.write('"""Cross-sell package"""\n')
    
    sys.path.append("./cross_sell_auto")

# Create data directory if needed
os.makedirs("data", exist_ok=True)

# Create a simple test input file
if not os.path.exists("data/input.csv"):
    with open("data/input.csv", "w") as f:
        f.write("id,name,email,phone,postal_code,has_health_insurance,vehicle_age,annual_mileage,score\n")
        f.write("1,John Doe,john@example.com,555-1234,12345,true,3,12000,0.85\n")
        f.write("2,Jane Smith,jane@example.com,555-5678,23456,true,5,8000,0.72\n")
        f.write("3,Bob Johnson,bob@example.com,555-9012,34567,true,1,15000,0.91\n")
    print("Created sample input data file")

try:
    # Import functions from the DAG
    print("\nImporting DAG functions...")
    from cross_sell_dag import (
        validate_environment, 
        process_csv_file,
        generate_predictions,
        send_leads_to_api,
        generate_daily_report,
        prepare_email_content
    )
    print("Successfully imported DAG functions")
    
    # Create context
    today = datetime.now().strftime('%Y-%m-%d')
    context = {
        'ds': today,
        'task_instance': SimpleMockTaskInstance(),
        'dag_run': type('obj', (object,), {'run_id': f'test_run_{today}'})
    }
    print(f"Created test context for date: {today}")
    
    # Run the functions in sequence
    print("\n1. Running validate_environment")
    validate_environment(**context)
    
    print("\n2. Running process_csv_file")
    process_csv_file(**context)
    
    print("\n3. Running generate_predictions")
    generate_predictions(**context)
    
    print("\n4. Running send_leads_to_api")
    send_leads_to_api(**context)
    
    print("\n5. Running generate_daily_report")
    generate_daily_report(**context)
    
    print("\n6. Running prepare_email_content")
    email_content = prepare_email_content(**context)
    print("Email preview (first 3 lines):")
    print("\n".join(email_content.split("\n")[:3]))
    
    # Success
    print("\n===== SIMPLE TEST COMPLETED SUCCESSFULLY =====")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 