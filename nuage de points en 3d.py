# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:41:32 2024

@author: hugop
"""

import pandas as pd
import matplotlib.pyplot as plt




chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)


donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
donnees['Runtime'] = donnees['Runtime'].replace(' min', '', regex=True).astype(float)


colonnes_quantitatives = ['Rating', 'Votes', 'Gross', 'Metascore', 'Runtime']
donnees = donnees[colonnes_quantitatives]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


x = donnees['Votes']
y = donnees['Gross']
z = donnees['Metascore']

ax.scatter(x, y, z, c='r', marker='o')


ax.set_xlabel('Votes')
ax.set_ylabel('Gross')
ax.set_zlabel('Metascore')


plt.title('Nuage de points 3D - Corrélation entre Votes, Gross et Metascore')
plt.show()