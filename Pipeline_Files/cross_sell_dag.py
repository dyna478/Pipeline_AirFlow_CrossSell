"""
Airflow DAG for Cross-sell Automobile Insurance Workflow
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import logging
import json
import os
import sys
import joblib  # or pickle, or tensorflow, etc.

# Add project directory to Python path
sys.path.append('/opt/airflow/dags/cross_sell_auto')

# Load Telus API credentials from Airflow Variables
os.environ['TELUS_API_KEY'] = Variable.get('TELUS_API_KEY', '')
os.environ['TELUS_USE_TEST'] = Variable.get('TELUS_USE_TEST', 'true')

from config import AIRFLOW_CONFIG, validate_config
from main_runner import CrossSellProcessor

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(AIRFLOW_CONFIG.START_DATE_DAYS_AGO),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': AIRFLOW_CONFIG.RETRIES,
    'retry_delay': timedelta(minutes=AIRFLOW_CONFIG.RETRY_DELAY_MINUTES),
    'email': ['data-team@yourcompany.com']  # Update with your email
}

# Create DAG
dag = DAG(
    AIRFLOW_CONFIG.DAG_ID,
    default_args=default_args,
    description='Cross-sell automobile insurance to health insurance clients',
    schedule_interval=AIRFLOW_CONFIG.SCHEDULE_INTERVAL,
    max_active_runs=AIRFLOW_CONFIG.MAX_ACTIVE_RUNS,
    catchup=AIRFLOW_CONFIG.CATCHUP,
    tags=['cross-sell', 'insurance', 'automobile', 'telus-api']
)

def validate_environment(**context):
    """
    Validate that all required environment variables and configurations are set
    """
    try:
        logger.info("Validating environment configuration")
        
        # Validate configuration
        validate_config()
        
        # Check if input file exists
        from config import FILE_CONFIG
        if not os.path.exists(FILE_CONFIG.INPUT_CSV_PATH):
            raise FileNotFoundError(f"Input CSV file not found: {FILE_CONFIG.INPUT_CSV_PATH}")
        
        # Test API connectivity
        from telus_api_client import TelusAPIClient
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
        
        # Load the model
        model_path = '/path/to/your/saved/model.pkl'
        model = joblib.load(model_path)

        # Use the model for predictions
        predictions = model.predict(data)
        
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
        report_path = f"/opt/airflow/reports/cross_sell_report_{context['ds']}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
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
        
        SUCCESS RATE: {results.get('successful_sends', 0)}/{results.get('leads_sent', 0)} 
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
        
        # Add summary statistics if available
        summary = results.get('summary', {})
        if summary:
            email_content += f"\nDETAILED STATISTICS:\n"
            email_content += "-" * 20 + "\n"
            email_content += f"• Conversion rate: {summary.get('conversion_rate', 0):.1f}%\n"
            
            # Top cities
            top_cities = summary.get('top_cities', {})
            if top_cities:
                email_content += f"\nTop Cities:\n"
                for city, count in list(top_cities.items())[:5]:
                    email_content += f"  - {city}: {count}\n"
        
        return email_content
        
    except Exception as e:
        logger.error(f"Error preparing email content: {str(e)}")
        return f"Error generating report: {str(e)}"

# Define tasks
validate_env_task = PythonOperator(
    task_id='validate_environment',
    python_callable=validate_environment,
    dag=dag,
)

process_csv_task = PythonOperator(
    task_id='process_csv_file',
    python_callable=process_csv_file,
    dag=dag,
)

generate_predictions_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag,
)

send_leads_task = PythonOperator(
    task_id='send_leads_to_api',
    python_callable=send_leads_to_api,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_daily_report',
    python_callable=generate_daily_report,
    dag=dag,
)

# Email task for daily report
email_task = EmailOperator(
    task_id='send_daily_report_email',
    to=['data-team@yourcompany.com'],  # Update with your email
    subject='Cross-sell Automobile Insurance - Daily Report - {{ ds }}',
    html_content="{{ task_instance.xcom_pull(task_ids='prepare_email_content') }}",
    dag=dag,
)

prepare_email_task = PythonOperator(
    task_id='prepare_email_content',
    python_callable=prepare_email_content,
    dag=dag,
)

# Cleanup task
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='''
    # Clean up temporary files older than 7 days
    find /opt/airflow/logs/cross_sell* -type f -mtime +7 -delete 2>/dev/null || true
    find /opt/airflow/reports/cross_sell* -type f -mtime +30 -delete 2>/dev/null || true
    echo "Cleanup completed"
    ''',
    dag=dag,
)

# Set task dependencies
validate_env_task >> process_csv_task
process_csv_task >> [generate_predictions_task, send_leads_task]
[generate_predictions_task, send_leads_task] >> generate_report_task
generate_report_task >> prepare_email_task >> email_task
email_task >> cleanup_task