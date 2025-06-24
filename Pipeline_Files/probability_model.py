"""
Probability model for automobile insurance cross-sell
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class ProbabilityModel:
    """
    Model for predicting automobile insurance cross-sell probability
    """
    
    def __init__(self):
        self.config = MODEL_CONFIG
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Load model if it exists
        if self.config.MODEL_PATH and os.path.exists(self.config.MODEL_PATH):
            self.load_model(self.config.MODEL_PATH)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the model
        Adjust this method based on your actual data structure
        """
        try:
            # Create a copy to avoid modifying original data
            features_df = df.copy()
            
            # Calculate age if birth_date is available
            if 'birth_date' in features_df.columns:
                features_df['birth_date'] = pd.to_datetime(features_df['birth_date'], errors='coerce')
                features_df['age'] = (pd.Timestamp.now() - features_df['birth_date']).dt.days / 365.25
            elif 'age' not in features_df.columns:
                # Default age if not available
                features_df['age'] = 35
            
            # Gender encoding
            if 'gender' in features_df.columns:
                features_df['gender_encoded'] = features_df['gender'].map({'M': 1, 'F': 0}).fillna(0.5)
            else:
                features_df['gender_encoded'] = 0.5
            
            # City size estimation (you might want to use a proper city database)
            if 'city' in features_df.columns:
                features_df['city_size'] = self._estimate_city_size(features_df['city'])
            else:
                features_df['city_size'] = 2  # Medium city default
            
            # Health premium normalization
            if 'health_premium' in features_df.columns:
                features_df['health_premium'] = pd.to_numeric(features_df['health_premium'], errors='coerce').fillna(0)
                features_df['health_premium_log'] = np.log1p(features_df['health_premium'])
            else:
                features_df['health_premium'] = 500  # Default premium
                features_df['health_premium_log'] = np.log1p(500)
            
            # Years with company
            if 'contract_start_date' in features_df.columns:
                features_df['contract_start_date'] = pd.to_datetime(features_df['contract_start_date'], errors='coerce')
                features_df['years_with_company'] = (pd.Timestamp.now() - features_df['contract_start_date']).dt.days / 365.25
            elif 'years_with_company' not in features_df.columns:
                features_df['years_with_company'] = 2  # Default
            
            # Claims history (if available)
            if 'claims_count' not in features_df.columns:
                features_df['claims_count'] = 0
            
            # Select final features
            self.feature_columns = [
                'age', 'gender_encoded', 'city_size', 'health_premium_log',
                'years_with_company', 'claims_count'
            ]
            
            # Ensure all feature columns exist
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            return features_df[self.feature_columns]
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def _estimate_city_size(self, cities: pd.Series) -> pd.Series:
        """
        Estimate city size category
        0: Small, 1: Medium, 2: Large
        """
        # This is a simplified approach - in practice, you'd use a proper city database
        major_cities = ['casablanca', 'rabat', 'marrakech', 'fes', 'tangier', 'agadir', 'meknes', 'oujda']
        
        def categorize_city(city):
            if pd.isna(city):
                return 1  # Medium default
            city_lower = str(city).lower()
            if any(major in city_lower for major in major_cities):
                return 2  # Large
            elif len(city_lower) > 8:  # Arbitrary rule for medium cities
                return 1  # Medium
            else:
                return 0  # Small
        
        return cities.apply(categorize_city)
    
    def train_model(self, df: pd.DataFrame, target_column: str = 'has_auto_insurance') -> Dict[str, Any]:
        """
        Train the probability model
        This is a placeholder - replace with your actual training logic
        """
        try:
            logger.info("Training automobile cross-sell model...")
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Target variable (you need to create this based on your business logic)
            if target_column not in df.columns:
                # Create synthetic target for demonstration
                # In practice, this should be based on historical data
                y = self._create_synthetic_target(X)
                logger.warning("Using synthetic target variable - replace with real data")
            else:
                y = df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Model trained - Train score: {train_score:.3f}, Test score: {test_score:.3f}")
            
            self.is_trained = True
            
            return {
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _create_synthetic_target(self, X: pd.DataFrame) -> pd.Series:
        """
        Create synthetic target variable for demonstration
        Replace this with your actual business logic
        """
        # Simple heuristic: higher probability for:
        # - Older people
        # - Higher health premiums
        # - Longer relationship with company
        # - Larger cities
        
        np.random.seed(42)
        
        prob = (
            0.3 * (X['age'] / X['age'].max()) +
            0.25 * (X['health_premium_log'] / X['health_premium_log'].max()) +
            0.25 * (X['years_with_company'] / X['years_with_company'].max()) +
            0.2 * (X['city_size'] / 2)
        )
        
        # Add some noise
        prob += np.random.normal(0, 0.1, len(prob))
        prob = np.clip(prob, 0, 1)
        
        # Convert to binary target
        return (prob > 0.6).astype(int)
    
    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cross-sell probability for clients
        
        Returns:
            np.ndarray: Probability scores
        """
        try:
            if not self.is_trained and self.model is None:
                raise ValueError("Model is not trained. Please train the model first.")
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
            
            logger.info(f"Generated predictions for {len(probabilities)} clients")
            logger.info(f"Mean probability: {np.mean(probabilities):.3f}")
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting probabilities: {str(e)}")
            raise
    
    def filter_by_threshold(self, df: pd.DataFrame, probabilities: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Filter clients by probability threshold
        
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: (filtered_df, filtered_probabilities)
        """
        try:
            # Create mask for clients above threshold
            mask = probabilities >= self.config.PROBABILITY_THRESHOLD
            
            # Filter data
            filtered_df = df[mask].copy()
            filtered_probabilities = probabilities[mask]
            
            logger.info(f"Filtered {len(filtered_df)} clients above threshold {self.config.PROBABILITY_THRESHOLD}")
            
            return filtered_df, filtered_probabilities
            
        except Exception as e:
            logger.error(f"Error filtering by threshold: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.encoders = model_data['encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise