import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run_glm(df_original, selected_features, glm_type):
    try:
        # Créer une copie pour le traitement
        df = df_original.copy()

        # Encoder les variables catégorielles
        encoders = {}
        for feature in selected_features:
            if pd.api.types.is_object_dtype(df[feature]):
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
                encoders[feature] = encoder

        # Sélectionner X et y
        X = df[selected_features]
        y = df['ClaimNb']

        # Standardiser les features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
        X_scaled = sm.add_constant(X_scaled)

        # Choisir le modèle
        family = {
            "poisson": sm.families.Poisson(),
            "binomial": sm.families.Binomial(),
        }.get(glm_type.lower())

        if not family:
            return "Type de modèle invalide"

        # Fit du modèle
        glm_model = sm.GLM(y, X_scaled, family=family).fit()

        return glm_model
    except Exception as e:
        return f"Erreur dans le modèle : {str(e)}"

# Interface Streamlit
st.title("Analyse GLM avec Standardisation et Exploration des Corrélations")

file = st.file_uploader("Télécharger le fichier CSV", type="csv")

if file:
    try:
        # Lecture du CSV
        df_original = pd.read_csv(file)

        # Conversion en string pour l'affichage Streamlit
        df_display = df_original.astype(str)

        # Affichage des données
        st.write("Aperçu des données :")
        st.write(df_display.head())

        # Sélection des features
        features = [col for col in df_original.columns if col != 'ClaimNb']
        selected_features = st.multiselect(
            "Choisir les features",
            features,
            default=[f for f in ['Exposure', 'Power', 'CarAge'] if f in features]
        )

        # Type de modèle
        glm_type = st.selectbox("Type de modèle GLM", ["Poisson", "Binomial"])

        if st.button("Analyser"):
            if selected_features:
                with st.spinner("Analyse en cours..."):
                    model = run_glm(df_original, selected_features, glm_type)
                    if isinstance(model, str):
                        st.error(model)
                    else:
                        st.text_area("Résultats", model.summary().as_text(), height=500)
                        
                        # Affichage des p-values
                        st.write("### P-Values des coefficients")
                        st.write(model.pvalues)

                        # Graphique des coefficients
                        st.write("### Impact des features")
                        coefs = pd.Series(model.params.values, index=model.params.index)
                        fig, ax = plt.subplots()
                        sns.barplot(x=coefs.index, y=coefs.values, ax=ax)
                        plt.xticks(rotation=45)
                        plt.title("Impact des features sur ClaimNb")
                        plt.xlabel("Feature")
                        plt.ylabel("Coefficient")
                        st.pyplot(fig)

                        # Matrice de corrélation
                        st.write("### Matrice de corrélation")
                        correlation_matrix = df_original[selected_features + ['ClaimNb']].corr()
                        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap="Blues", ax=ax_corr)
                        st.pyplot(fig_corr)
            else:
                st.warning("Veuillez sélectionner au moins une feature.")

    except Exception as e:
        st.error(f"Erreur : {str(e)}")
