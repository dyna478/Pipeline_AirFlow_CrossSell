"""
Classes de modèles utilisées pour la prédiction de cross-sell automobile
"""
import numpy as np

class DummyModel:
    """
    Classe de modèle factice qui peut être utilisée pour les tests 
    ou comme fallback si le modèle réel n'est pas disponible
    """
    def __init__(self):
        self.name = "Automobile Insurance Cross-Sell Model"
        self.version = "1.0"
        self.features = []
        
    def set_features(self, features):
        """Définir les caractéristiques utilisées par le modèle"""
        self.features = features
        
    def predict(self, X):
        """
        Prédire la propension à acheter une assurance automobile
        
        Args:
            X : array-like ou DataFrame contenant les caractéristiques
            
        Returns:
            array: Scores de propension entre 0 et 1
        """
        # Logique de prédiction simplifiée pour les tests
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        # Générer des prédictions aléatoires avec un biais vers le haut
        return np.random.beta(7, 3, size=n_samples)
    
    def predict_proba(self, X):
        """
        Prédire les probabilités de classes [non, oui]
        
        Args:
            X : array-like ou DataFrame contenant les caractéristiques
            
        Returns:
            array: Tableau de probabilités [P(non), P(oui)]
        """
        # Générer des probabilités
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        probs = np.random.beta(7, 3, size=n_samples)
        # Retourner les probabilités pour chaque classe [non, oui]
        return np.column_stack((1-probs, probs))
    
    def score(self, X, y):
        """
        Calculer le score du modèle
        
        Args:
            X : Caractéristiques
            y : Valeurs réelles
            
        Returns:
            float: Score
        """
        return 0.85  # Score fictif élevé pour les tests
        
    def __str__(self):
        return f"{self.name} (version {self.version})" 