# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:19:27 2024

@author: hugop
"""

import pandas as pd
import statsmodels.api as sm

# Chemin du fichier CSV
chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'

# Charger les données depuis le fichier CSV
donnees = pd.read_csv(chemin_fichier)

# Afficher les types de données de chaque colonne
print("Types de données avant nettoyage :")
print(donnees.dtypes)

# Afficher les valeurs uniques de chaque colonne de type 'object'
object_columns = donnees.select_dtypes(include=['object']).columns
for col in object_columns:
    print("Valeurs uniques de la colonne", col, ":", donnees[col].unique())

# Nettoyage des données
# Supprimer les lignes avec des valeurs manquantes ou non numériques
donnees_propres = donnees.dropna()  # Supprimer les lignes avec des valeurs manquantes
donnees_propres = donnees_propres.apply(pd.to_numeric, errors='coerce').dropna()  # Convertir les colonnes en numériques et supprimer les valeurs non numériques

# Sélectionner les colonnes pour l'analyse
X = donnees_propres[['Adventure', 'Comedy']]
y = donnees_propres['IMDB_Rating']

# Ajouter une colonne de constante pour l'intercept
X = sm.add_constant(X)

# Effectuer une régression linéaire
model = sm.OLS(y, X).fit()

# Afficher les résultats de la régression
print("\nRésultats de la régression linéaire :")
print(model.summary())

