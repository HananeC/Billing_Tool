import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Charger le modèle sauvegardé
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('best_xgb_model.json')
    return model

model = load_model()

# Titre de l'application
st.title("Prédiction du nombre de sinistres en assurance")

# Formulaire pour saisir les nouvelles données
st.header("Saisissez les informations du client")

car_age = st.slider("Age du véhicule (années)", min_value=0, max_value=30, value=5)
driver_age = st.slider("Age du conducteur (années)", min_value=18, max_value=90, value=35)
exposure = st.number_input("Exposition (fraction de l'année)", min_value=0.0, max_value=1.0, value=0.5)
density = st.number_input("Densité (population/km²)", min_value=0.0, max_value=5000.0, value=1000.0)

# Variables qualitatives avec toutes les options possibles
power_options = ['e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
brand_options = [
    'Japanese (except Nissan) or Korean',
    'Mercedes, Chrysler or BMW',
    'Opel, General Motors or Ford',
    'Renault, Nissan or Citroen',
    'Volkswagen, Audi, Skoda or Seat',
    'other'
]
region_options = [
    'Basse-Normandie',
    'Bretagne',
    'Centre',
    'Haute-Normandie',
    'Ile-de-France',
    'Limousin',
    'Nord-Pas-de-Calais',
    'Pays-de-la-Loire',
    'Poitou-Charentes'
]

power = st.selectbox("Catégorie de puissance", power_options)
brand = st.selectbox("Marque du véhicule", brand_options)
gas = st.selectbox("Type de carburant", ['Regular', 'Diesel'])
region = st.selectbox("Région", region_options)

def prepare_data(car_age, driver_age, exposure, density, power, brand, gas, region):
    # Création d'un DataFrame avec les nouvelles données
    data = {
        'Exposure': [exposure],
        'CarAge': [car_age],
        'DriverAge': [driver_age],
        'Density': [density],
        'Power': [power],
        'Brand': [brand],
        'Gas': [gas],
        'Region': [region]
    }
    df = pd.DataFrame(data)

    # Normalisation des variables numériques
    scaler = StandardScaler()
    numeric_cols = ['Exposure', 'CarAge', 'DriverAge', 'Density']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Création de toutes les colonnes dummy possibles
    # Power
    power_dummies = pd.get_dummies(df['Power'], prefix='Power')
    for p in power_options:
        if f'Power_{p}' not in power_dummies.columns:
            power_dummies[f'Power_{p}'] = 0

    # Brand
    brand_dummies = pd.get_dummies(df['Brand'], prefix='Brand')
    for b in brand_options:
        if f'Brand_{b}' not in brand_dummies.columns:
            brand_dummies[f'Brand_{b}'] = 0

    # Gas
    gas_dummies = pd.get_dummies(df['Gas'], prefix='Gas')
    if 'Gas_Regular' not in gas_dummies.columns:
        gas_dummies['Gas_Regular'] = 0

    # Region
    region_dummies = pd.get_dummies(df['Region'], prefix='Region')
    for r in region_options:
        if f'Region_{r}' not in region_dummies.columns:
            region_dummies[f'Region_{r}'] = 0

    # Combiner toutes les colonnes
    df_final = pd.concat([
        df[numeric_cols],
        power_dummies,
        brand_dummies,
        gas_dummies,
        region_dummies
    ], axis=1)

    # S'assurer que les colonnes sont dans le même ordre que lors de l'entraînement
    expected_columns = model.get_booster().feature_names
    df_final = df_final[expected_columns]

    return df_final

# Bouton pour lancer la prédiction
if st.button("Prédire le nombre de sinistres"):
    try:
        # Préparer les données pour la prédiction
        new_data = prepare_data(car_age, driver_age, exposure, density, power, brand, gas, region)
        
        # Faire la prédiction
        prediction = model.predict(new_data)
        prediction_proba = model.predict_proba(new_data)

        # Afficher les résultats
        st.subheader("Résultat de la prédiction")
        st.write(f"Nombre de sinistres prédit : {prediction[0]}")

        st.subheader("Probabilités pour chaque classe")
        st.write(f"Probabilité de 0 sinistre : {prediction_proba[0][0]:.2f}")
        st.write(f"Probabilité de 1 sinistre : {prediction_proba[0][1]:.2f}")
        st.write(f"Probabilité de 2 sinistres ou plus : {prediction_proba[0][2]:.2f}")
        
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction : {str(e)}")