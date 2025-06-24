#!/usr/bin/env python
"""
Test script for Cross-sell workflow without Airflow
"""
import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cross_sell_test")

print("===== TEST SCRIPT STARTED =====")

# Create mock classes for Airflow
class Variable:
    """Mock for Airflow Variable class"""
    _vars = {
        'TELUS_API_KEY': 'test_api_key',
        'TELUS_USE_TEST': 'true'
    }
    
    @classmethod
    def get(cls, key, default=''):
        print(f"Getting variable: {key}")
        return cls._vars.get(key, default)

class TaskInstance:
    """Mock for Airflow TaskInstance"""
    def __init__(self):
        self._xcom_data = {}
    
    def xcom_push(self, key, value):
        print(f"XCom Push: {key}")
        self._xcom_data[key] = value
        return True
    
    def xcom_pull(self, task_ids=None, key=None):
        print(f"XCom Pull: {key}")
        return self._xcom_data.get(key)

# Print directory structure for debugging
print("Current directory:", os.getcwd())
print("Contents:")
for item in os.listdir("."):
    path = os.path.join(".", item)
    if os.path.isdir(path):
        print(f"  DIR: {item}")
    else:
        print(f"  FILE: {item}")

# Add project directory to path
project_path = './cross_sell_auto'
sys.path.append(project_path)
print(f"Added {project_path} to Python path")

# Setup environment variables
os.environ['TELUS_API_KEY'] = Variable.get('TELUS_API_KEY', '')
os.environ['TELUS_USE_TEST'] = Variable.get('TELUS_USE_TEST', 'true')
print("Environment variables set")

def run_test():
    """Run the cross-sell workflow without Airflow"""
    try:
        print("\n=== STARTING CROSS-SELL TEST RUN ===")
        
        # Create test context
        today = datetime.now().strftime('%Y-%m-%d')
        context = {
            'ds': today,
            'task_instance': TaskInstance(),
            'dag_run': type('obj', (object,), {'run_id': f'test_run_{today}'})
        }
        print(f"Created test context for date: {today}")
        
        # Import the functions from the DAG 
        print("Attempting to import from cross_sell_dag...")
        try:
            from cross_sell_dag import (
                validate_environment, 
                process_csv_file,
                generate_predictions,
                send_leads_to_api,
                generate_daily_report,
                prepare_email_content
            )
            print("Successfully imported functions from cross_sell_dag")
        except ImportError as e:
            print(f"ERROR: Could not import from cross_sell_dag: {str(e)}")
            return False
        
        # Run each task
        print("\nRunning task: validate_environment")
        validate_environment(**context)
        
        print("\nRunning task: process_csv_file")
        process_csv_file(**context)
        
        print("\nRunning task: generate_predictions")
        generate_predictions(**context)
        
        print("\nRunning task: send_leads_to_api")
        send_leads_to_api(**context)
        
        print("\nRunning task: generate_daily_report")
        generate_daily_report(**context)
        
        print("\nRunning task: prepare_email_content")
        email_content = prepare_email_content(**context)
        print("Email content preview (first few lines):")
        for line in email_content.split("\n")[:5]:
            print(f"  {line}")
        
        # Get final results
        results = context['task_instance'].xcom_pull(key='processing_results')
        if results:
            print("\n=== TEST RUN RESULTS ===")
            print(f"Records processed: {results.get('original_records', 0)}")
            print(f"Qualified clients: {results.get('qualified_clients', 0)}")
            print(f"Leads sent: {results.get('leads_sent', 0)}")
        
        print("\n=== CROSS-SELL TEST RUN COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    print(f"\nTest completed with {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1) 