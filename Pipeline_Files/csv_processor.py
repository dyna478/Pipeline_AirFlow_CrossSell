"""
CSV processor for health insurance client data
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from config import FILE_CONFIG

logger = logging.getLogger(__name__)

class CSVProcessor:
    """
    Processes CSV files containing health insurance client data
    """
    
    def __init__(self):
        self.config = FILE_CONFIG
        self.processed_clients = set()
        self._load_processed_clients()
    
    def _load_processed_clients(self):
        """Load list of previously processed clients"""
        try:
            if os.path.exists(self.config.PROCESSED_CSV_PATH):
                processed_df = pd.read_parquet(self.config.PROCESSED_CSV_PATH)
                if 'client_id' in processed_df.columns:
                    self.processed_clients = set(processed_df['client_id'].astype(str))
                    logger.info(f"Loaded {len(self.processed_clients)} previously processed clients")
                elif 'telephone' in processed_df.columns:
                    # Use telephone as unique identifier if client_id not available
                    self.processed_clients = set(processed_df['telephone'].astype(str))
                    logger.info(f"Loaded {len(self.processed_clients)} previously processed clients (by phone)")
        except Exception as e:
            logger.warning(f"Could not load processed clients: {str(e)}")
            self.processed_clients = set()
    
    def read_parquet_file(self, filepath: str) -> pd.DataFrame:
        """
        Read and validate Parquet file
        """
        try:
            logger.info(f"Reading Parquet file: {filepath}")
            df = pd.read_parquet(filepath)
            logger.info(f"Parquet loaded: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error reading Parquet file: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean the data
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: (cleaned_df, validation_issues)
        """
        try:
            issues = []
            cleaned_df = df.copy()
            
            # Check for required columns
            required_columns = ['telephone']  # Minimum required
            optional_columns = ['nom', 'prenom', 'ville', 'date_creation', 'client_id']
            
            missing_required = [col for col in required_columns if col not in cleaned_df.columns]
            if missing_required:
                issues.append(f"Missing required columns: {missing_required}")
                raise ValueError(f"Missing required columns: {missing_required}")
            
            # Log missing optional columns
            missing_optional = [col for col in optional_columns if col not in cleaned_df.columns]
            if missing_optional:
                issues.append(f"Missing optional columns: {missing_optional}")
                logger.warning(f"Missing optional columns: {missing_optional}")
            
            # Clean telephone numbers
            if 'telephone' in cleaned_df.columns:
                original_count = len(cleaned_df)
                # Remove all non-digit characters
                cleaned_df['telephone'] = cleaned_df['telephone'].astype(str).str.replace(r'[^\d]', '', regex=True)
                # Normalize numbers: remove leading country code if present (e.g., 212)
                cleaned_df['telephone'] = cleaned_df['telephone'].apply(
                    lambda x: x[-10:] if (len(x) > 10 and x[-10:].startswith(('05', '06', '07'))) else x
                )
                # Validate Moroccan phone numbers: start with 05, 06, or 07 and have exactly 10 digits
                valid_phone_mask = cleaned_df['telephone'].str.match(r'^0[5-7][0-9]{8}$')
                cleaned_df = cleaned_df[valid_phone_mask]
                
                if len(cleaned_df) < original_count:
                    removed_count = original_count - len(cleaned_df)
                    issues.append(f"Removed {removed_count} rows with invalid Moroccan phone numbers")
                    logger.warning(f"Removed {removed_count} rows with invalid Moroccan phone numbers")
            
            # Parse date column if available
            if self.config.DATE_COLUMN in cleaned_df.columns:
                try:
                    cleaned_df[self.config.DATE_COLUMN] = pd.to_datetime(
                        cleaned_df[self.config.DATE_COLUMN], 
                        format=self.config.DATE_FORMAT,
                        errors='coerce'
                    )
                    
                    # Remove rows with invalid dates
                    valid_date_mask = cleaned_df[self.config.DATE_COLUMN].notna()
                    invalid_dates = (~valid_date_mask).sum()
                    if invalid_dates > 0:
                        issues.append(f"Found {invalid_dates} rows with invalid dates")
                        cleaned_df = cleaned_df[valid_date_mask]
                        
                except Exception as e:
                    issues.append(f"Error parsing dates: {str(e)}")
                    logger.error(f"Error parsing dates: {str(e)}")
            
            # Clean text fields
            text_columns = ['nom', 'prenom', 'ville']
            for col in text_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                    cleaned_df[col] = cleaned_df[col].replace('nan', '')
            
            # Remove duplicates based on phone number
            original_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['telephone'], keep='first')
            
            if len(cleaned_df) < original_count:
                removed_duplicates = original_count - len(cleaned_df)
                issues.append(f"Removed {removed_duplicates} duplicate entries")
                logger.info(f"Removed {removed_duplicates} duplicate entries")
            
            logger.info(f"Data validation completed: {len(cleaned_df)} valid rows remaining")
            
            return cleaned_df, issues
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise
    
    def filter_new_clients(self, df: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter for new clients that have not been processed yet using a 'processed' flag.
        Adds the column if it does not exist.
        Args:
            df: DataFrame with client data
            target_date: (Unused in this approach)
        Returns:
            pd.DataFrame: Filtered DataFrame with new clients only
        """
        try:
            # Ensure 'processed' column exists
            if 'processed' not in df.columns:
                df['processed'] = False
            else:
                df['processed'] = df['processed'].fillna(False)

            # Only select unprocessed clients
            new_clients_df = df[df['processed'] == False].copy()
            logger.info(f"Filtered {len(new_clients_df)} unprocessed clients using 'processed' flag.")
            return new_clients_df
        except Exception as e:
            logger.error(f"Error filtering new clients: {str(e)}")
            raise

    def mark_clients_as_processed(self, df: pd.DataFrame, processed_clients: pd.DataFrame):
        """
        Mark the given clients as processed in the main DataFrame and save back to Parquet.
        Args:
            df: The main DataFrame (all clients)
            processed_clients: DataFrame of clients that were just processed
        """
        try:
            if 'client_id' in df.columns and 'client_id' in processed_clients.columns:
                df.loc[df['client_id'].isin(processed_clients['client_id']), 'processed'] = True
            elif 'telephone' in df.columns and 'telephone' in processed_clients.columns:
                df.loc[df['telephone'].isin(processed_clients['telephone']), 'processed'] = True
            else:
                logger.warning("No suitable identifier found to mark clients as processed.")
            # Save the updated DataFrame back to Parquet
            df.to_parquet(self.config.INPUT_CSV_PATH, index=False)
            logger.info(f"Marked {len(processed_clients)} clients as processed and updated Parquet file.")
        except Exception as e:
            logger.error(f"Error marking clients as processed: {str(e)}")
            raise
    
    def save_processed_clients(self, df: pd.DataFrame):
        """
        Save processed clients to avoid reprocessing
        """
        try:
            if len(df) == 0:
                logger.info("No clients to save")
                return
            # Prepare data for saving
            save_df = df.copy()
            save_df['processed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Select relevant columns
            columns_to_save = ['processed_date']
            if 'client_id' in save_df.columns:
                columns_to_save.append('client_id')
            if 'telephone' in save_df.columns:
                columns_to_save.append('telephone')
            if 'nom' in save_df.columns:
                columns_to_save.append('nom')
            if 'prenom' in save_df.columns:
                columns_to_save.append('prenom')
            save_df = save_df[columns_to_save]
            # Append to existing file or create new one
            if os.path.exists(self.config.PROCESSED_CSV_PATH):
                existing_df = pd.read_parquet(self.config.PROCESSED_CSV_PATH)
                combined_df = pd.concat([existing_df, save_df], ignore_index=True)
                combined_df.to_parquet(self.config.PROCESSED_CSV_PATH, index=False)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config.PROCESSED_CSV_PATH), exist_ok=True)
                save_df.to_parquet(self.config.PROCESSED_CSV_PATH, index=False)
            # Update in-memory set
            if 'client_id' in save_df.columns:
                self.processed_clients.update(save_df['client_id'].astype(str))
            elif 'telephone' in save_df.columns:
                self.processed_clients.update(save_df['telephone'].astype(str))
            logger.info(f"Saved {len(save_df)} processed clients to {self.config.PROCESSED_CSV_PATH}")
        except Exception as e:
            logger.error(f"Error saving processed clients: {str(e)}")
            raise
    
    def generate_summary_report(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame, 
                              qualified_df: pd.DataFrame, sent_df: pd.DataFrame) -> Dict:
        """
        Generate a summary report of the processing
        """
        try:
            report = {
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_records': len(original_df),
                'new_clients': len(filtered_df),
                'qualified_clients': len(qualified_df),
                'leads_sent': len(sent_df),
                'success_rate': len(sent_df) / len(qualified_df) * 100 if len(qualified_df) > 0 else 0,
                'conversion_rate': len(qualified_df) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            }
            
            # Add demographic breakdown if available
            if 'ville' in filtered_df.columns:
                report['top_cities'] = filtered_df['ville'].value_counts().head(5).to_dict()
            
            if 'sexe' in filtered_df.columns or 'gender' in filtered_df.columns:
                gender_col = 'sexe' if 'sexe' in filtered_df.columns else 'gender'
                report['gender_distribution'] = filtered_df[gender_col].value_counts().to_dict()
            
            logger.info(f"Generated summary report: {report}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_old_processed_files(self, days_to_keep: int = 30):
        """
        Clean up old processed client records to prevent file from growing too large
        """
        try:
            if not os.path.exists(self.config.PROCESSED_CSV_PATH):
                return
            # Read existing processed file
            processed_df = pd.read_parquet(self.config.PROCESSED_CSV_PATH)
            if 'processed_date' in processed_df.columns:
                # Convert to datetime
                processed_df['processed_date'] = pd.to_datetime(processed_df['processed_date'])
                # Filter to keep only recent records
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                recent_df = processed_df[processed_df['processed_date'] >= cutoff_date]
                if len(recent_df) < len(processed_df):
                    # Save cleaned up file
                    recent_df.to_parquet(self.config.PROCESSED_CSV_PATH, index=False)
                    removed_count = len(processed_df) - len(recent_df)
                    logger.info(f"Cleaned up {removed_count} old processed client records")
                    # Update in-memory set
                    if 'client_id' in recent_df.columns:
                        self.processed_clients = set(recent_df['client_id'].astype(str))
                    elif 'telephone' in recent_df.columns:
                        self.processed_clients = set(recent_df['telephone'].astype(str))
        except Exception as e:
            logger.error(f"Error cleaning up processed files: {str(e)}")