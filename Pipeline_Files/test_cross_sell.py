#!/usr/bin/env python
"""
Test script for Cross-sell workflow without Airflow
"""
import os
import sys
import json
import logging
from datetime import datetime

# Set up more visible logging - CRITICAL CHANGE FOR DEBUGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to stdout
    ]
)
logger = logging.getLogger("cross_sell_test")

# Print statement for debugging
print("SCRIPT STARTED - SETTING UP ENVIRONMENT")

try:
    # Import mock classes
    from mock_classes import Variable, DAGContext
    print("Successfully imported mock classes")
except ImportError as e:
    print(f"ERROR importing mock classes: {str(e)}")
    sys.exit(1)

# Set up environment to use our mocks
try:
    sys.modules['airflow.models'] = type('MockAirflowModels', (), {'Variable': Variable})
    print("Successfully set up mock for airflow.models")
except Exception as e:
    print(f"ERROR setting up mocks: {str(e)}")
    sys.exit(1)

# Add project directory to path - update with your actual path
project_path = './cross_sell_auto'
sys.path.append(project_path)
print(f"Added {project_path} to Python path")

# Setup environment variables from mock variables 
os.environ['TELUS_API_KEY'] = Variable.get('TELUS_API_KEY', '')
os.environ['TELUS_USE_TEST'] = Variable.get('TELUS_USE_TEST', 'true')
print(f"Set environment variables: TELUS_API_KEY={os.environ.get('TELUS_API_KEY')} (masked), TELUS_USE_TEST={os.environ.get('TELUS_USE_TEST')}")

# Print working directory and list files for debugging
print(f"Current working directory: {os.getcwd()}")
print("Files in current directory:")
for item in os.listdir("."):
    if os.path.isdir(item):
        print(f"  {item}/")
    else:
        print(f"  {item}")

def run_test():
    """Run the cross-sell workflow without Airflow"""
    try:
        print("=== STARTING CROSS-SELL TEST RUN ===")
        
        # Create DAG context with all necessary Airflow variables
        context = DAGContext().get_context()
        print("Created DAG context successfully")
        
        # Import the functions from the DAG 
        # Do this after setting up mocks to ensure they're used
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
            print(f"ERROR importing from cross_sell_dag: {str(e)}")
            print(f"Python path: {sys.path}")
            return False
        
        # 1. Validate environment
        print("Step 1: Validating environment")
        validate_environment(**context)
        
        # 2. Process CSV file
        print("Step 2: Processing CSV file")
        process_csv_file(**context)
        
        # 3. Generate predictions
        print("Step 3: Generating predictions")
        generate_predictions(**context)
        
        # 4. Send leads to API
        print("Step 4: Sending leads to API")
        send_leads_to_api(**context)
        
        # 5. Generate daily report
        print("Step 5: Generating daily report")
        report_path = generate_daily_report(**context)
        print(f"Report saved to: {report_path}")
        
        # 6. Prepare email content
        print("Step 6: Preparing email content")
        email_content = prepare_email_content(**context)
        print("Email Content Preview:")
        print("-" * 50)
        for line in email_content.split("\n")[:10]:  # Show first 10 lines
            print(line)
        print("-" * 50)
        
        # Get final results
        results = context['task_instance'].xcom_pull(key='processing_results')
        if results:
            print("=== TEST RUN RESULTS ===")
            print(json.dumps(results, indent=2))
        
        print("=== CROSS-SELL TEST RUN COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting test execution...")
    success = run_test()
    print(f"Test completed with {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1) 