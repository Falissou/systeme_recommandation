import streamlit as st 
import numpy as np 
from utils import load_clean_movie_data, get_recom

# Chargement des données
movie_data = load_clean_movie_data("data/Movie.csv")

# Titre de la page
st.title('Application de Recommandation de Films')

# Input pour le choix du film
name = st.selectbox('Choisissez un film', movie_data['title'].unique())

# Input pour le nombre de recommandations
num_recommandations = st.number_input('Nombre de films à recommander', min_value=1, value=10, step=1)

# Bouton pour obtenir les recommandations
if st.button('Obtenir les recommandations'):
    try:
        # Chargement de la matrice de similarité
        similarity_matrix_load = np.load('models/similarity_matrix.npy')
        
        # Utilisation de la fonction de recommandation
        recommandations = get_recom(
            titre=name,
            data=movie_data,
            sim=similarity_matrix_load,
            count=num_recommandations
        )
        
        # Affichage des résultats
        if recommandations:
            st.write('Films recommandés :')
            for rec in recommandations:
                st.markdown(f"- {rec}")
        else:
            st.warning("Aucune recommandation disponible pour ce film.")
    
    except FileNotFoundError as e:
        st.error("Erreur : Fichier de la matrice de similarité introuvable.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {str(e)}")
