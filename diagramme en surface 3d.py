# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:40:41 2024

@author: hugop
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)


donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
donnees['Runtime'] = donnees['Runtime'].replace(' min', '', regex=True).astype(float)


colonnes_quantitatives = ['Rating', 'Votes', 'Gross', 'Metascore', 'Runtime']
donnees = donnees[colonnes_quantitatives]

donnees = donnees.dropna()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


z = donnees['Rating']
x = donnees['Votes']
y = donnees['Runtime']


surface = ax.plot_trisurf(x, y, z, cmap='twilight_shifted', edgecolor='k', linewidth=0.2)


ax.set_zlabel('Rating')
ax.set_xlabel('Votes')
ax.set_ylabel('Runtime')


fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

plt.title('Diagramme en surface 3D - Corrélation entre Votes, Gross et Metascore')
plt.show()