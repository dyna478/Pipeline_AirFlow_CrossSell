"""
Modified version of DAG for testing without Airflow
"""
from datetime import datetime, timedelta
import logging
import json
import os
import sys
import pickle  # or joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project directory to Python path if needed
if './cross_sell_auto' not in sys.path:
    sys.path.append('./cross_sell_auto')

# Set environment variables if not already set
if 'TELUS_API_KEY' not in os.environ:
    os.environ['TELUS_API_KEY'] = 'test_key_for_testing'
if 'TELUS_USE_TEST' not in os.environ:
    os.environ['TELUS_USE_TEST'] = 'true'

# Import config and other modules
try:
    from cross_sell_auto.config import AIRFLOW_CONFIG, FILE_CONFIG, validate_config
    from cross_sell_auto.main_runner import CrossSellProcessor
    from cross_sell_auto.telus_api_client import TelusAPIClient
except ImportError as e:
    logger.error(f"Import error: {str(e)}. Creating dummy modules...")
    
    # Create dummy config
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
        MODEL_PATH = "./models/model.pkl"
        
    def validate_config():
        """Dummy validate_config"""
        logger.info("Validating config...")
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./data/output", exist_ok=True)
        
        # Create test data if not exists
        if not os.path.exists(FILE_CONFIG.INPUT_CSV_PATH):
            with open(FILE_CONFIG.INPUT_CSV_PATH, 'w') as f:
                f.write("id,name,email,phone,postal_code,has_health_insurance,vehicle_age,annual_mileage,score\n")
                f.write("1,John Doe,john@example.com,555-1234,12345,true,3,12000,0.85\n")
                f.write("2,Jane Smith,jane@example.com,555-5678,23456,true,5,8000,0.72\n")
        return True
    
    # Create dummy processor
    class CrossSellProcessor:
        def __init__(self):
            logger.info("Initializing CrossSellProcessor")
            
        def process_daily_file(self, target_date=None):
            logger.info(f"Processing file for date: {target_date}")
            return {
                "processing_date": target_date,
                "original_records": 100,
                "new_clients": 50,
                "qualified_clients": 25,
                "leads_sent": 20,
                "successful_sends": 18,
                "errors": ["Test error"],
                "summary": {"conversion_rate": 72.0}
            }
    
    # Create dummy API client
    class TelusAPIClient:
        def __init__(self):
            logger.info("Initializing TelusAPIClient")
            
        def test_connection(self):
            logger.info("Testing API connection")
            return True


# DAG functions extracted from the original file

def validate_environment(**context):
    """
    Validate that all required environment variables and configurations are set
    """
    try:
        logger.info("Validating environment configuration")
        
        # Validate configuration
        validate_config()
        
        # Check if input file exists
        if not os.path.exists(FILE_CONFIG.INPUT_CSV_PATH):
            raise FileNotFoundError(f"Input CSV file not found: {FILE_CONFIG.INPUT_CSV_PATH}")
        
        # Test API connectivity
        api_client = TelusAPIClient()
        if not api_client.test_connection():
            raise Exception("Cannot connect to TelusHolding API")
        
        logger.info("Environment validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        raise

def process_csv_file(**context):
    """
    Process the daily CSV file for new health insurance clients
    """
    try:
        logger.info("Starting CSV file processing")
        
        # Get execution date
        execution_date = context['ds']  # YYYY-MM-DD format
        
        # Initialize processor
        processor = CrossSellProcessor()
        
        # Process the file
        results = processor.process_daily_file(target_date=execution_date)
        
        # Store results in XCom for downstream tasks
        context['task_instance'].xcom_push(key='processing_results', value=results)
        
        logger.info(f"CSV processing completed: {json.dumps(results, indent=2)}")
        
        return results
        
    except Exception as e:
        logger.error(f"CSV processing failed: {str(e)}")
        raise

def generate_predictions(**context):
    """
    Generate cross-sell probability predictions for new clients
    """
    try:
        logger.info("Generating cross-sell predictions")
        
        # This step is integrated into the main processing
        # but can be separated if needed for monitoring
        results = context['task_instance'].xcom_pull(key='processing_results')
        
        if not results:
            logger.warning("No processing results found")
            return
        
        qualified_clients = results.get('qualified_clients', 0)
        logger.info(f"Generated predictions for {qualified_clients} qualified clients")
        
        # Load the model if it exists
        try:
            model_path = FILE_CONFIG.MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded from {model_path}")
                
                # Just test the model works
                sample_prediction = model.predict([1, 2, 3])
                logger.info(f"Sample prediction: {sample_prediction}")
            else:
                logger.warning(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
        
        return qualified_clients
        
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise

def send_leads_to_api(**context):
    """
    Send qualified leads to TelusHolding API
    """
    try:
        logger.info("Sending leads to TelusHolding API")
        
        # This step is also integrated into main processing
        # but separated here for better monitoring and error handling
        results = context['task_instance'].xcom_pull(key='processing_results')
        
        if not results:
            logger.warning("No processing results found")
            return
        
        leads_sent = results.get('leads_sent', 0)
        successful_sends = results.get('successful_sends', 0)
        errors = results.get('errors', [])
        
        logger.info(f"API sending completed: {successful_sends}/{leads_sent} successful")
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during API sending")
            for error in errors:
                logger.error(f"API Error: {error}")
        
        return {
            'leads_sent': leads_sent,
            'successful_sends': successful_sends,
            'error_count': len(errors)
        }
        
    except Exception as e:
        logger.error(f"API sending failed: {str(e)}")
        raise

def generate_daily_report(**context):
    """
    Generate daily processing report
    """
    try:
        logger.info("Generating daily report")
        
        # Get results from previous tasks
        results = context['task_instance'].xcom_pull(key='processing_results')
        
        if not results:
            logger.warning("No processing results found for report")
            return
        
        # Create report
        report = {
            'date': context['ds'],
            'dag_run_id': context['dag_run'].run_id,
            'execution_time': datetime.now().isoformat(),
            'results': results
        }
        
        # Save report to file
        os.makedirs("./reports", exist_ok=True)
        report_path = f"./reports/cross_sell_report_{context['ds']}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Store report in XCom for email task
        context['task_instance'].xcom_push(key='daily_report', value=report)
        
        logger.info(f"Daily report saved to {report_path}")
        
        return report_path
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise

def prepare_email_content(**context):
    """
    Prepare email content for daily report
    """
    try:
        results = context['task_instance'].xcom_pull(key='processing_results')
        
        if not results:
            return "No processing results available"
        
        # Create email content
        email_content = f"""
        Cross-sell Automobile Insurance - Daily Report
        =============================================
        
        Date: {context['ds']}
        Processing Time: {results.get('processing_date', 'Unknown')}
        
        SUMMARY:
        --------
        • Original records processed: {results.get('original_records', 0)}
        • New clients identified: {results.get('new_clients', 0)}
        • Clients qualified for cross-sell: {results.get('qualified_clients', 0)}
        • Leads sent to TelusHolding API: {results.get('leads_sent', 0)}
        • Successfully processed leads: {results.get('successful_sends', 0)}
        
        SUCCESS RATE: {results.get('successful_sends', 0)}/{results.get('leads_sent', 1)} 
        ({(results.get('successful_sends', 0)/results.get('leads_sent', 1)*100):.1f}%)
        """
        
        # Add errors if any
        errors = results.get('errors', [])
        if errors:
            email_content += f"\nERRORS ENCOUNTERED ({len(errors)}):\n"
            email_content += "-" * 30 + "\n"
            for i, error in enumerate(errors[:10], 1):  # Limit to first 10 errors
                email_content += f"{i}. {error}\n"
            
            if len(errors) > 10:
                email_content += f"\n... and {len(errors) - 10} more errors\n"
        
        logger.info("Email content prepared successfully")
        return email_content
        
    except Exception as e:
        logger.error(f"Error preparing email content: {str(e)}")
        return f"Error generating report: {str(e)}" 