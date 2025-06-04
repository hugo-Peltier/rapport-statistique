# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:37:49 2024

@author: hugop
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)


donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
donnees['Runtime'] = donnees['Runtime'].replace(' min', '', regex=True).astype(float)


colonnes_quantitatives = ['Rating', 'Votes', 'Gross', 'Metascore', 'Runtime']


palette = sns.color_palette("husl", n_colors=len(donnees['Genre'].unique()))


sns.pairplot(donnees, vars=colonnes_quantitatives, hue='Genre', palette=palette)
plt.suptitle('Matrice de Diagrammes en Nuage de Points avec Couleurs basées sur le Genre', y=1.02)
plt.show()
