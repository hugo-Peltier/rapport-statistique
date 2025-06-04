# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:08:16 2024

@author: hugop
"""
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, parallel_coordinates, radviz

#%% EXO 1



ville = ["Paris", "Marseille", "Lyon", "Toulouse", "Nice"]
population = [2187526, 863310, 516092, 479553, 340017]
superficie = [105.4, 240.6, 47.87, 118.3, 71.92]


Serie1 = pd.Series(population, index=ville, name="Population")
Serie2 = pd.Series(superficie, index=ville, name="Superficie")

print(Serie1)
print(Serie2)
#%%

population_lyon = Serie1["Lyon"]
superficie_lyon = Serie2["Lyon"]

print(f"La population de Lyon est {population_lyon}.")
print(f"La superficie de Lyon est {superficie_lyon} km².")

#%%



data = {
    "Ville": ville,
    "Population": population,
    "Superficie (km²)": superficie
}

df = pd.DataFrame(data)

print(df)

#%%
print("Trois premières lignes :")
print(df.head(3))
print("\nTrois dernières lignes :")
print(df.tail(3))

#%%
print("\nInformations sur les colonnes :")
print(df.info())

#%%
n_lignes, n_colonnes = df.shape
print(f"\nNombre de lignes : {n_lignes}, Nombre de colonnes : {n_colonnes}")

#%%
print("\nStatistiques récapitulatives :")
print(df.describe())

#%%
print("\n1ère colonne :")
print(df.iloc[:, 0])

#%%
print("\n2ème ligne :")
print(df.iloc[1])

#%%
print("\n2ème et 3ème lignes :")
print(df.iloc[1:3])

#%%
print("\n2ème colonne à partir de la 2ème ligne :")
print(df.iloc[1:, 1])  

#%%
print("\nVilles avec une population supérieure à 700 000 :")
print(df[df["Population"] > 700000])

#%%
print("\nVilles classées par superficie (décroissant) :")
print(df.sort_values(by="Superficie (km²)", ascending=False))

#%% EXO 3
from sklearn.datasets import load_iris


iris = load_iris()

#%%
print("Type de la variable iris :", type(iris))





























#%%
print("\nClés de iris :", iris.keys())
print("Target :", iris.target)
print("Target names :", iris.target_names)
print("Feature names :", iris.feature_names)
print("Données :", iris.data)
print("Description :", iris.DESCR)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for i in range(len(iris.target_names)):
    plt.hist(iris.data[iris.target == i, 0], alpha=0.5, label=iris.target_names[i])
plt.title('Histogramme de la longueur du sépale')
plt.xlabel('Longueur du sépale (cm)')
plt.ylabel('Fréquence')
plt.legend()
plt.show()

#%%
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target
print("\nDataFrame des données Iris :")
print(df_iris.head())

#%%
df_iris.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)', title='Scatter Plot')
plt.show()

df_iris.plot(kind='bar', title='Bar Plot of Sepal Lengths')
plt.show()

df_iris.plot(kind='hist', y='sepal length (cm)', title='Histogram of Sepal Lengths', bins=10)
plt.show()

#%%
plt.figure(figsize=(10, 6))
for i in range(len(iris.target_names)):
    plt.hist(iris.data[iris.target == i, 2], alpha=0.5, label=iris.target_names[i])
plt.title('Histogramme de la longueur du pétale')
plt.xlabel('Longueur du pétale (cm)')
plt.ylabel('Fréquence')
plt.legend()
plt.show()

#%%
def plot_histogram(col):
    plt.hist(df_iris[col], bins=10, alpha=0.7)
    plt.title(f'Histogramme de {col}')
    plt.xlabel(col)
    plt.ylabel('Fréquence')
    plt.show()

# Exemple d'utilisation
plot_histogram('sepal length (cm)')

#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, column in enumerate(df_iris.columns[:-1]):  
    ax = axes[i // 2, i % 2]
    df_iris[column].plot(kind='hist', ax=ax, bins=10, alpha=0.7)
    ax.set_title(f'Histogramme de {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Fréquence')

plt.tight_layout()
plt.show()

#%%
# Scatter matrix
from pandas.plotting import scatter_matrix, parallel_coordinates, radviz
scatter_matrix(df_iris, figsize=(10, 10), alpha=0.5)
plt.show()
#%%
# Boxplot
plt.figure(figsize=(10, 6))
df_iris.boxplot(column='sepal length (cm)', by='species')
plt.title('Boxplot de la longueur du sépale par espèce')
plt.suptitle('')
plt.show()
#%%
# Radviz
plt.figure(figsize=(10, 6))
radviz(df_iris, 'species')
plt.title('Radviz de l\'ensemble de données Iris')
plt.show()
#%%
# Histogramme
df_iris.hist(figsize=(10, 8))
plt.show()



#%%


#%%
matiere = pd.read_csv(r"C:\Users\hugop\OneDrive\Bureau\Matiere.csv", sep=";")
etudiant = pd.read_csv(r"C:\Users\hugop\OneDrive\Bureau\Etudiant.csv",sep=";")
tab_note = pd.read_csv(r"C:\Users\hugop\OneDrive\Bureau\TabNote.csv",sep=";")

#%%
print("Type de Matiere :", type(matiere))
print("Type de Etudiant :", type(etudiant))
print("Type de TabNote :", type(tab_note))

#%%
print("\nTableau TabNote :")
print(tab_note)
print("\nPropriétés de TabNote :")
print(tab_note.info())

#%%
print("\nRésumé statistique de TabNote :")
print(tab_note.describe())

#%%
print("\nNombre de valeurs manquantes dans TabNote :")
print(tab_note.isnull().sum())

#%%
tab_note.fillna(0, inplace=True)
print("\nValeurs manquantes remplacées par 0 dans TabNote :")
print(tab_note.isnull().sum())

#%%


algo_students = [6]  
AlgoTab = tab_note[tab_note['NumE'].isin(algo_students)][['NumE', 'Note']]
print("\nSous DataFrame AlgoTab :")
print(AlgoTab)

#%%
Algo = dict(zip(AlgoTab['NumE'], AlgoTab['Note']))
print("\nDictionnaire Algo :", Algo)

#%%exo10

Prenoms = ['Servane', 'Florence', 'Jordan','Christophe']
Noms = ['Coppalle', 'Mendes', 'Houari','Guerville']
AlgoTab.insert(1, 'Prenoms', Prenoms[:len(AlgoTab)])
AlgoTab.insert(2, 'Noms', Noms[:len(AlgoTab)])
#%%exoooo1
print(AlgoTab)

print("Question 11")
algo_dataframe = tab_note[tab_note['NumM'] == 2]
print(algo_dataframe)

stat_dataframe = tab_note[tab_note['NumM'] == 3]
print(stat_dataframe)

