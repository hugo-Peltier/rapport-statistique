# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:39:25 2024

@author: hugop
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)


donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
donnees['Runtime'] = donnees['Runtime'].replace(' min', '', regex=True).astype(float)


colonnes_quantitatives = ['Rating', 'Votes', 'Gross', 'Metascore', 'Runtime']

scatter_matrix(donnees[colonnes_quantitatives], alpha=0.8, figsize=(12, 12), diagonal='hist')
plt.suptitle('Matrice de Diagrammes en Nuage de Points', y=0.92)
plt.show()