# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Chemin vers votre fichier de données
data_path = 'Housing.csv'

# Charger les données
df = pd.read_csv(data_path)

# Prétraitement des données (exemple : séparation des features et de la cible)
X = df.drop('price', axis=1)  # Remplacer 'target_column_name' par le nom de votre colonne cible
y = df['price']

# Division des données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du modèle (exemple : Régression Linéaire)
model = LinearRegression()

# Entraînement du modèle
model.fit(X_train, y_train)

# Chemin pour sauvegarder le modèle
model_path = 'housing_price_prediction_model.pkl'

# Sauvegarder le modèle formé
joblib.dump(model, model_path)

print("Entraînement du modèle terminé et modèle sauvegardé avec succès.")
