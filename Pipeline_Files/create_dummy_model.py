#!/usr/bin/env python
"""
Ce script crée un modèle factice pour les tests
"""
import os
import pickle
import numpy as np
import pandas as pd
from model_classes import DummyModel

print("=== Création d'un modèle factice pour les tests ===")

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Créer une instance du modèle factice
model = DummyModel()

# Définir quelques caractéristiques types pour une assurance automobile
features = [
    "age", "income", "vehicle_age", "annual_mileage", 
    "has_health_insurance", "credit_score"
]
model.set_features(features)

# Enregistrer le modèle dans un fichier pickle
model_path = "models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Modèle factice créé et enregistré dans '{model_path}'")

# Optionnel: tester le modèle
print("\n=== Test du modèle ===")

# Créer quelques données de test
n_samples = 5
test_data = pd.DataFrame({
    "age": np.random.randint(18, 75, n_samples),
    "income": np.random.randint(30000, 150000, n_samples),
    "vehicle_age": np.random.randint(0, 15, n_samples),
    "annual_mileage": np.random.randint(5000, 25000, n_samples),
    "has_health_insurance": np.random.choice([0, 1], n_samples),
    "credit_score": np.random.randint(500, 850, n_samples)
})

# Faire des prédictions avec le modèle
predictions = model.predict(test_data)

print(f"\nDonnées de test ({n_samples} échantillons):")
print(test_data)

print("\nPrédictions:")
for i, pred in enumerate(predictions):
    print(f"Client {i+1}: {pred:.4f} ({pred >= 0.6 and 'Qualifié' or 'Non qualifié'})")

print("\n=== Terminé ===")

with open("chemin/vers/votre/modele.pkl", "rb") as f:
    model = pickle.load(f)
print(type(model).__name__)  # Affiche le nom de la classe principale 