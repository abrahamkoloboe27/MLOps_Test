# scripts/train_model.py

import pandas as pd
from pycaret.regression import create_model, setup, save_model



# Chemin vers votre fichier de données
data_path = 'data/Housing.csv' # Remplacer 'data_path' par le chemin de votre fichier de données

# Charger les données
df = pd.read_csv(data_path)

# Initialiser l'environnement PyCaret
s = setup(df, 
          target = 'price', 
          session_id = 123, 
          train_size=0.7,
          fold_strategy="kfold",
          fold=10,
          numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'],
          categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'],
          profile = False
          )

# Comparer les modèles

#best_model = compare_models(["ridge", "rf","knn","lasso","lr"],sort="rmse", n_select=3)
best_model = create_model("lr")

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
save_model(best_model, 'housing_price_prediction_model')
#save_model(final_model, 'housing_price_prediction_model')

# Copier housing_price_prediction_model.pkl dans le dossier API
# Obetenez le path actuel
import os
current_path = os.getcwd()
# Copier le fichier dans le dossier API
import shutil
shutil.copy('API/housing_price_prediction_model.pkl', current_path)

