"""
Configuration for Cross-sell Project
"""
import os

class AIRFLOW_CONFIG:
    DAG_ID = "cross_sell_automobile_insurance" 
    START_DATE_DAYS_AGO = 1
    SCHEDULE_INTERVAL = "@daily"
    MAX_ACTIVE_RUNS = 1
    CATCHUP = False
    RETRIES = 3
    RETRY_DELAY_MINUTES = 5

class FILE_CONFIG:
    # Utiliser le fichier de prédéfini pour les tests
    INPUT_CSV_PATH = "./data/pred_def_test_2025.csv"
    OUTPUT_PATH = "./data/output"
    MODEL_PATH = "./models/model.pkl"
    REPORT_PATH = "./reports"

def validate_config():
    """Validate configuration settings"""
    import os
    
    # Check if directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    os.makedirs(FILE_CONFIG.OUTPUT_PATH, exist_ok=True)
    
    # Vérifier si le fichier d'entrée existe
    if not os.path.exists(FILE_CONFIG.INPUT_CSV_PATH):
        print(f"ATTENTION: Fichier d'entrée non trouvé: {FILE_CONFIG.INPUT_CSV_PATH}")
        print("Veuillez vous assurer que le fichier pred_def_test_2025.csv existe dans le dossier data/")
        return False
    else:
        print(f"Fichier d'entrée trouvé: {FILE_CONFIG.INPUT_CSV_PATH}")
    
    # Check if model exists
    if not os.path.exists(FILE_CONFIG.MODEL_PATH):
        print(f"WARNING: Model file not found at {FILE_CONFIG.MODEL_PATH}")
        print("You can generate a dummy model by running 'python models/dummy_model.py'")
    
    print("Configuration validated successfully")
    return True 