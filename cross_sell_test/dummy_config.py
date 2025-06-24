"""
Configuration for Cross-sell Project
"""

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
    """Validate configuration settings"""
    import os
    
    # Check if directories exist
    if not os.path.isdir("./data"):
        os.makedirs("./data", exist_ok=True)
        
    if not os.path.isdir("./models"):
        os.makedirs("./models", exist_ok=True)
    
    # Create dummy input file if not exists
    if not os.path.exists(FILE_CONFIG.INPUT_CSV_PATH):
        with open(FILE_CONFIG.INPUT_CSV_PATH, 'w') as f:
            f.write("id,name,email,phone,postal_code,has_health_insurance,vehicle_age,annual_mileage,score\n")
            f.write("1,John Doe,john@example.com,555-1234,12345,true,3,12000,0.85\n")
            f.write("2,Jane Smith,jane@example.com,555-5678,23456,true,5,8000,0.72\n")
            f.write("3,Bob Johnson,bob@example.com,555-9012,34567,true,1,15000,0.91\n")
    
    # Create output directory if not exists
    if not os.path.isdir(FILE_CONFIG.OUTPUT_PATH):
        os.makedirs(FILE_CONFIG.OUTPUT_PATH, exist_ok=True)
    
    return True 