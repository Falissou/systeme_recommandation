import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_clean_movie_data(movie_file):
    """
    Charge et nettoie les données des films.
    
    Parameters:
    - movie_file : str, chemin vers le fichier CSV des films.
    
    Returns:
    - DataFrame : Données nettoyées.
    """
    try:
        data = pd.read_csv(movie_file)
        data.dropna(subset=['title'], inplace=True)  # S'assurer que les titres ne sont pas manquants
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {movie_file} est introuvable.")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement des données : {str(e)}")

def get_recom(titre, data, sim, count=10):
    """
    Obtenir des recommandations basées sur la similarité cosinus.
    
    Parameters:
    - titre : str, titre du film pour lequel des recommandations sont recherchées.
    - data : DataFrame, contenant au moins une colonne 'title'.
    - sim : matrice de similarité (numpy array ou DataFrame).
    - count : int, nombre de recommandations souhaitées (par défaut 10).
    
    Returns:
    - list : titres des films recommandés.
    """
    # Trouver l'indice correspondant au titre
    index = data[data['title'].str.lower() == titre.lower()].index
    
    # Vérifier si le titre existe
    if len(index) == 0:
        return []
    
    # Récupérer l'indice
    idx = index[0]
    
    # Vérifier si l'indice est dans les limites de la matrice
    if idx >= len(sim):
        return []
    
    # Obtenir les similarités de la ligne correspondante
    similarities = list(enumerate(sim[idx]))
    
    # Trier par score de similarité décroissant
    recommandations = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Obtenir les meilleurs résultats, en excluant l'élément lui-même
    top_rec = recommandations[1:count + 1]
    
    # Générer la liste des titres
    titres = []
    for rec in top_rec:
        rec_idx = rec[0]  # Indice de la recommandation
        if rec_idx < len(data):  # Vérifier les limites
            titre_rec = data.iloc[rec_idx]['title']
            titres.append(titre_rec)
    
    return titres
