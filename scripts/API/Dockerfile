# Utilisez l'image officielle de Python comme image de base
FROM python:3.11-slim

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les fichiers de dépendances vers le conteneur
COPY requirements_.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements_.txt

# Copier tous les fichiers du dossier API dans le conteneur 
COPY scripts/API /app

# Copier le modèle dans le conteneur
COPY housing_price_prediction_model.pkl /app

# Exposez le port sur lequel l'application va tourner
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
