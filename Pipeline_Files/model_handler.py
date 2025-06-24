import joblib
import logging
from typing import Optional, Any
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handler for model operations using joblib serialization"""
    
    def __init__(self, model_path: str = 'models/model.joblib'):
        """
        Initialize the model handler
        
        Args:
            model_path: Path where the model will be saved/loaded
        """
        self.model_path = model_path
        self.model = None
        self._ensure_model_directory()
    
    def _ensure_model_directory(self):
        """Ensure the model directory exists"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def save_model(self, model: Any, metadata: Optional[dict] = None) -> bool:
        """
        Save the model using joblib
        
        Args:
            model: The model to save
            metadata: Optional dictionary with model metadata
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare metadata
            save_metadata = {
                'saved_at': datetime.now().isoformat(),
                'model_type': str(type(model).__name__),
                **metadata if metadata else {}
            }
            
            # Save both model and metadata
            joblib.dump({
                'model': model,
                'metadata': save_metadata
            }, self.model_path)
            
            logger.info(f"Model successfully saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self) -> tuple[Optional[Any], Optional[dict]]:
        """
        Load the model using joblib
        
        Returns:
            tuple: (model, metadata) if successful, (None, None) if failed
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"No model found at {self.model_path}")
                return None, None
            
            # Load the model and metadata
            loaded_data = joblib.load(self.model_path)
            self.model = loaded_data['model']
            metadata = loaded_data['metadata']
            
            logger.info(f"Model successfully loaded from {self.model_path}")
            logger.info(f"Model metadata: {metadata}")
            
            return self.model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None
    
    def predict(self, X):
        """
        Make predictions using the loaded model
        
        Args:
            X: Input features for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            self.model, _ = self.load_model()
            if self.model is None:
                raise ValueError("No model loaded and couldn't load from file")
        
        return self.model.predict(X)
    
    def get_model(self) -> Optional[Any]:
        """
        Get the current model instance
        
        Returns:
            The model if loaded, None otherwise
        """
        if self.model is None:
            self.model, _ = self.load_model()
        return self.model 