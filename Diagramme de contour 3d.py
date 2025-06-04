import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata


chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)


donnees['Votes'] = pd.to_numeric(donnees['Votes'].str.replace(',', ''))


def convert_gross(value):
    if pd.notna(value):
        if 'M' in str(value):
            return float(str(value).replace('M', '').replace('$', '')) * 1e6
        elif 'B' in str(value):
            return float(str(value).replace('B', '').replace('$', '')) * 1e9
        else:
            return float(str(value).replace('$', '').replace(',', ''))
    return value

donnees['Gross'] = donnees['Gross'].apply(convert_gross)

donnees['Gross'].fillna(0, inplace=True)


x = donnees['Votes'].values
y = donnees['Rating'].values

X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))


Z = griddata((x, y), donnees['Gross'].values, (X, Y), method='cubic')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
contour = ax.contour3D(X, Y, Z, 50, cmap='inferno')


fig.colorbar(contour)


ax.set_xlabel('Votes')
ax.set_ylabel('Rating')
ax.set_zlabel('Gross')

plt.show()
