# scripts/train_model.py



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import os

# Chemin vers votre fichier de données
data_path = 'data/Housing.csv'  # Remplacer 'data_path' par le chemin de votre fichier de données

# Charger les données
df = pd.read_csv(data_path)

# Séparer les caractéristiques et la cible
X = df.drop(columns='price')
y = df['price']

# Séparer les colonnes numériques et catégorielles
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Préparer le préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Initialiser les modèles à comparer
models = {
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(),
    'KNeighbors': KNeighborsRegressor(),
    'Lasso': Lasso(),
    'LinearRegression': LinearRegression()
}

# Initialiser la validation croisée
kf = KFold(n_splits=10, shuffle=True, random_state=123)

# Comparer les modèles
# Initialize variables for best model selection
best_model_name = None
best_model = None
best_score = float('inf')

for name, model in models.items():
    # Créer un pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Calculer les scores avec validation croisée
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    mean_score = -scores.mean()
    
    print(f'{name} RMSE: {mean_score}')
    
    # Sélectionner le meilleur modèle
    if mean_score < best_score:
        best_score = mean_score
        best_model_name = name
        best_model = pipeline

# Entraîner le meilleur modèle sur l'ensemble d'entraînement complet
best_model.fit(X, y)

# Sauvegarder le meilleur modèle
current_path = os.getcwd()
model_path = os.path.join(current_path, 'housing_price_prediction_model.pkl')
joblib.dump(best_model, model_path)

print(f'Best model: {best_model_name} with RMSE: {best_score}')
print(f'Model saved to: {model_path}')



#import pandas as pd
#from pycaret.regression import create_model, setup, save_model
#import joblib


# Chemin vers votre fichier de données
#data_path = 'data/Housing.csv' # Remplacer 'data_path' par le chemin de votre fichier de données

# Charger les données
#df = pd.read_csv(data_path)

# Initialiser l'environnement PyCaret
# s = setup(df, 
#           target = 'price', 
#           session_id = 123, 
#           train_size=0.7,
#           fold_strategy="kfold",
#           fold=10,
#           numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'],
#           categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'],
#           profile = False
#           )

# Comparer les modèles

#best_model = compare_models(["ridge", "rf","knn","lasso","lr"],sort="rmse", n_select=3)
#best_model = create_model("lr")

# Highlight the minimum value of each column
#highlight_min = lambda x: ['background-color: yellow' if v == x.min() else '' for v in x]
#pull().style.apply(highlight_min, axis=0)

# Fine-tune the best model
#tuned_models = [tune_model(model) for model in best_model]
#tuned_models


# Compare the tuned models
#best_model_tuned = compare_models(tuned_models, sort="rmse", n_select=3)


#Finalize the best model
#final_model = finalize_model(best_model_tuned[0])
#save_model(best_model, 'housing_price_prediction_model')
#save_model(final_model, 'housing_price_prediction_model')

# Copier housing_price_prediction_model.pkl dans le dossier API
# Obetenez le path actuel
import os
current_path = os.getcwd()
print(current_path)
#joblib.dump(best_model, os.path.join(current_path, 'housing_price_prediction_model.pkl'))


print("Success")