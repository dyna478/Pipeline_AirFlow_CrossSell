#!/usr/bin/env python
"""
Simple test runner for cross_sell DAG without Airflow
This script uses test_cross_sell_dag.py which has no Airflow dependencies
"""
import os
import sys
import json
from datetime import datetime

print("===== TESTING CROSS-SELL DAG WITHOUT AIRFLOW =====")

# Create directories
print("Setting up directories...")
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Set environment variables
os.environ['TELUS_API_KEY'] = 'fake_test_key_123'
os.environ['TELUS_USE_TEST'] = 'true'

# Create a simple task instance mock
class MockTaskInstance:
    def __init__(self):
        self._data = {}
    
    def xcom_push(self, key, value):
        print(f"Storing data: {key}")
        self._data[key] = value
        
    def xcom_pull(self, key=None, task_ids=None):
        print(f"Retrieving data: {key}")
        return self._data.get(key)

# Create sample model file
try:
    print("Creating test model file...")
    if not os.path.exists("models/model.pkl"):
        import pickle
        
        class SimpleModel:
            def predict(self, data):
                print(f"Predicting on data: {data}")
                if hasattr(data, '__iter__'):
                    return [1] * len(data) 
                return 1
        
        model = SimpleModel()
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Test model created")
except Exception as e:
    print(f"Warning: Could not create model file: {str(e)}")

# Import the functions
try:
    print("\nImporting functions from test_cross_sell_dag...")
    from test_cross_sell_dag import (
        validate_environment,
        process_csv_file,
        generate_predictions,
        send_leads_to_api,
        generate_daily_report,
        prepare_email_content
    )
    print("Successfully imported functions")
except Exception as e:
    print(f"ERROR: Failed to import functions: {str(e)}")
    sys.exit(1)

# Run the test
try:
    print("\nRunning cross-sell workflow...")
    
    # Set up context
    today = datetime.now().strftime('%Y-%m-%d')
    test_context = {
        'ds': today,
        'task_instance': MockTaskInstance(),
        'dag_run': type('obj', (object,), {'run_id': f'test_run_{today}'})
    }
    
    # Step 1: Validate environment
    print("\n-- Step 1: Validating environment")
    validate_environment(**test_context)
    
    # Step 2: Process CSV file
    print("\n-- Step 2: Processing CSV file")
    process_csv_file(**test_context)
    
    # Step 3: Generate predictions
    print("\n-- Step 3: Generating predictions")
    generate_predictions(**test_context)
    
    # Step 4: Send leads to API
    print("\n-- Step 4: Sending leads to API")
    api_results = send_leads_to_api(**test_context)
    print(f"API results: {api_results}")
    
    # Step 5: Generate report
    print("\n-- Step 5: Generating report")
    report_path = generate_daily_report(**test_context)
    print(f"Report saved to: {report_path}")
    
    # Step 6: Prepare email content
    print("\n-- Step 6: Preparing email content")
    email = prepare_email_content(**test_context)
    print("\nEmail preview:")
    print("-------------")
    preview_lines = email.split('\n')[:10]
    print('\n'.join(preview_lines))
    print("-------------")
    
    print("\n===== TEST COMPLETED SUCCESSFULLY =====")
    
except Exception as e:
    print(f"\nERROR: Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 