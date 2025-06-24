"""
Main runner script for the cross-sell automobile insurance workflow
"""
import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import json
import os

# Import our modules
from config import validate_config, FILE_CONFIG, MODEL_CONFIG, TELUS_CONFIG
from csv_processor import CSVProcessor
from probability_model import ProbabilityModel
from telus_api_client import TelusAPIClient

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(FILE_CONFIG.LOG_PATH), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(FILE_CONFIG.LOG_PATH),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

class CrossSellProcessor:
    """Main processor for cross-sell automobile insurance workflow"""
    
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.model = ProbabilityModel()
        self.api_client = TelusAPIClient()
        self.results = {
            'processing_date': datetime.now().isoformat(),
            'original_records': 0,
            'new_clients': 0,
            'qualified_clients': 0,
            'leads_sent': 0,
            'successful_sends': 0,
            'errors': []
        }
    
    def process_daily_file(self, csv_file_path: Optional[str] = None, 
                          target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Process daily CSV file for cross-sell opportunities
        
        Args:
            csv_file_path: Path to CSV file. If None, uses config default.
            target_date: Target date for filtering (YYYY-MM-DD). If None, uses today.
            
        Returns:
            Dict with processing results
        """
        try:
            logger.info("Starting cross-sell automobile insurance processing")
            
            # Validate configuration
            validate_config()
            
            # Use default path if not provided
            if csv_file_path is None:
                csv_file_path = FILE_CONFIG.INPUT_CSV_PATH
            
            # Test API connection first
            if not self.api_client.test_connection():
                raise Exception("Cannot connect to TelusHolding API")
            
            # Step 1: Read and validate CSV
            logger.info("Step 1: Reading and validating Parquet data")
            df = self.csv_processor.read_parquet_file(csv_file_path)
            self.results['original_records'] = len(df)
            
            # Validate and clean data
            df_clean, validation_issues = self.csv_processor.validate_data(df)
            self.results['validation_issues'] = validation_issues
            
            # Step 2: Filter for new clients
            logger.info("Step 2: Filtering for new clients")
            df_new = self.csv_processor.filter_new_clients(df_clean, target_date)
            self.results['new_clients'] = len(df_new)
            
            if len(df_new) == 0:
                logger.info("No new clients to process")
                return self.results
            
            # Step 3: Generate probabilities
            logger.info("Step 3: Generating cross-sell probabilities")
            probabilities = self.model.predict_probability(df_new)
            
            # Add probabilities to dataframe for logging
            df_new = df_new.copy()
            df_new['cross_sell_probability'] = probabilities
            
            # Step 4: Filter by ²
            logger.info(f"Step 4: Filtering by threshold ({MODEL_CONFIG.PROBABILITY_THRESHOLD})")
            df_qualified, qualified_probs = self.model.filter_by_threshold(df_new, probabilities)
            self.results['qualified_clients'] = len(df_qualified)
            
            if len(df_qualified) == 0:
                logger.info("No clients qualified for cross-sell")
                return self.results
            
            # Step 5: Send leads to API
            logger.info("Step 5: Sending qualified leads to TelusHolding API")
            df_sent = self._send_leads_to_api(df_qualified, qualified_probs)
            self.results['leads_sent'] = len(df_qualified)
            self.results['successful_sends'] = len(df_sent)
            
            # Step 6: Save processed clients
            logger.info("Step 6: Saving processed clients")
            self.csv_processor.save_processed_clients(df_new)
            
            # Step 7: Generate summary report
            logger.info("Step 7: Generating summary report")
            summary = self.csv_processor.generate_summary_report(
                df, df_new, df_qualified, df_sent
            )
            self.results['summary'] = summary
            
            # Clean up old processed files
            self.csv_processor.cleanup_old_processed_files()
            
            logger.info("Cross-sell processing completed successfully")
            logger.info(f"Results: {json.dumps(self.results, indent=2)}")
            
            return self.results
            
        except Exception as e:
            error_msg = f"Error in cross-sell processing: {str(e)}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            raise
    
    def _send_leads_to_api(self, df_qualified: pd.DataFrame, probabilities) -> pd.DataFrame:
        """
        Send qualified leads to TelusHolding API
        
        Returns:
            pd.DataFrame: Successfully sent leads
        """
        sent_leads = []
        
        for idx, (_, row) in enumerate(df_qualified.iterrows()):
            try:
                # Convert row to dictionary
                client_data = row.to_dict()
                client_data['cross_sell_probability'] = probabilities[idx]
                
                logger.info(f"Sending lead for client: {client_data.get('telephone', 'Unknown')}")
                
                # Send to API
                success, message, response = self.api_client.insert_lead(client_data)
                
                if success:
                    sent_leads.append(row)
                    logger.info(f"Lead sent successfully: {message}")
                else:
                    error_msg = f"Failed to send lead: {message}"
                    logger.error(error_msg)
                    self.results['errors'].append(error_msg)
                
            except Exception as e:
                error_msg = f"Error sending lead for client {client_data.get('telephone', 'Unknown')}: {str(e)}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)
        
        # Return DataFrame of successfully sent leads
        if sent_leads:
            return pd.DataFrame(sent_leads)
        else:
            return pd.DataFrame()
    
    def train_model_if_needed(self, training_data_path: Optional[str] = None):
        """
        Train or retrain the probability model if needed
        """
        try:
            if not self.model.is_trained and training_data_path:
                logger.info("Training probability model")
                
                # Load training data
                training_df = pd.read_parquet(training_data_path)
                
                # Train model
                training_results = self.model.train_model(training_df)
                
                # Save model
                model_path = MODEL_CONFIG.MODEL_PATH or '/opt/airflow/models/cross_sell_model.joblib'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save_model(model_path)
                
                logger.info(f"Model training completed: {training_results}")
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

def main():
    """Main entry point for the script"""
    setup_logging()
    
    try:
        # Initialize processor
        processor = CrossSellProcessor()
        
        # Check if we need to train the model
        # processor.train_model_if_needed()  # Uncomment if you have training data
        
        # Process daily file
        results = processor.process_daily_file()
        
        # Print results
        print("\nProcessing Results:")
        print("=" * 50)
        print(f"Original records: {results['original_records']}")
        print(f"New clients: {results['new_clients']}")
        print(f"Qualified clients: {results['qualified_clients']}")
        print(f"Leads sent: {results['leads_sent']}")
        print(f"Successful sends: {results['successful_sends']}")
        
        if results['errors']:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        return results
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
