"""
Configuration file for the cross-sell automobile insurance workflow
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TelusAPIConfig:
    """Configuration for TelusHolding API"""
    # API endpoints
    PROD_INSERT_URL: str = "http://192.168.144.10/API/lead_insertV2.php"
    TEST_INSERT_URL: str = "https://telusholding.cloud/LEADSMANAGER/API/lead_insertV2_SAND.php"
    
    # Authentication credentials (should be set via environment variables)
    API_KEY: str = os.getenv('TELUS_API_KEY', '')
    
    # Use test environment by default
    USE_TEST_ENV: bool = os.getenv('TELUS_USE_TEST', 'true').lower() == 'true'
    
    @property
    def insert_url(self) -> str:
        return self.TEST_INSERT_URL if self.USE_TEST_ENV else self.PROD_INSERT_URL

@dataclass
class ModelConfig:
    """Configuration for the probability model"""
    # Threshold for sending leads to API
    PROBABILITY_THRESHOLD: float = float(os.getenv('PROBABILITY_THRESHOLD', '0.7'))
    
    # Model parameters (adjust based on your actual model)
    MODEL_FEATURES: list = None
    MODEL_PATH: Optional[str] = os.getenv('MODEL_PATH', None)
    
    def __post_init__(self):
        if self.MODEL_FEATURES is None:
            self.MODEL_FEATURES = [
                'age', 'gender', 'city', 'health_premium', 
                'years_with_company', 'claims_history'
            ]

@dataclass
class FileConfig:
    """Configuration for file processing"""
    # CSV file paths
    INPUT_CSV_PATH: str = os.getenv('INPUT_CSV_PATH', '/opt/airflow/data/health_clients.csv')
    PROCESSED_CSV_PATH: str = os.getenv('PROCESSED_CSV_PATH', '/opt/airflow/data/processed_clients.csv')
    LOG_PATH: str = os.getenv('LOG_PATH', '/opt/airflow/logs/cross_sell.log')
    
    # Date column name in CSV
    DATE_COLUMN: str = os.getenv('DATE_COLUMN', 'date_creation')
    DATE_FORMAT: str = os.getenv('DATE_FORMAT', '%Y-%m-%d')

@dataclass
class AirflowConfig:
    """Configuration for Airflow DAG"""
    DAG_ID: str = 'cross_sell_automobile_insurance'
    SCHEDULE_INTERVAL: str = '@daily'  # Run daily
    START_DATE_DAYS_AGO: int = 1
    MAX_ACTIVE_RUNS: int = 1
    CATCHUP: bool = False
    
    # Task configuration
    RETRIES: int = 2
    RETRY_DELAY_MINUTES: int = 5

# Global configuration instances
TELUS_CONFIG = TelusAPIConfig()
MODEL_CONFIG = ModelConfig()
FILE_CONFIG = FileConfig()
AIRFLOW_CONFIG = AirflowConfig()

# Validation
def validate_config():
    """Validate that all required configuration is present"""
    errors = []
    
    if not TELUS_CONFIG.API_KEY:
        errors.append("TELUS_API_KEY environment variable is required")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return True