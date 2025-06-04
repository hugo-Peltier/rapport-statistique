# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:35:58 2024

@author: hugop
"""

#Relation entre 2 variables : 
import pandas as pd
import matplotlib.pyplot as plt


chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)

moyenne = donnees['Rating'].mean()
mediane = donnees['Rating'].median()
mode = donnees['Rating'].mode()
ecart_type = donnees['Rating'].std()

print(f"Moyenne des notes: {moyenne}")
print(f"Médiane des notes: {mediane}")
print(f"Mode des notes: {mode.values}")
print(f"Ecart type des notes: {ecart_type}")

intervalles = [i/100 for i in range(750, 900, 25)]  
labels = [f'{i:.2f}-{i+0.25:.2f}' for i in intervalles[:-1]]

donnees['Intervalles'] = pd.cut(donnees['Rating'], bins=intervalles, labels=labels, include_lowest=True)

quantiles = [0.25, 0.5, 0.75]
valeurs_quantiles = donnees['Rating'].quantile(quantiles)

for q, valeur in zip(quantiles, valeurs_quantiles):
    print(f"Quantile de {q}: {valeur}")

intervalles_counts = donnees['Intervalles'].value_counts().sort_index()
intervalles = intervalles_counts.index
frequences = intervalles_counts.values

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

axes[0].bar(intervalles, frequences, color='pink', edgecolor='blue', width=0.8)
axes[0].set_title('Fréquence en fonction des intervalles de notes')
axes[0].set_xlabel('Intervalles de Notes')
axes[0].set_ylabel('Fréquence')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

pourcentages_effectifs = (frequences / len(donnees)) * 100
axes[1].pie(pourcentages_effectifs, labels=intervalles, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[1].set_title('Pourcentage des effectifs en fonction des intervalles de notes')

plt.tight_layout()
plt.show()
