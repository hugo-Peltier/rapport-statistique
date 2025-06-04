import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr









def menu_principal():
    nom_utilisateur = input("Entrez votre nom d'utilisateur : ")
    print(f"Bonjour {nom_utilisateur}, bienvenue dans notre projet d'étude des films d'animations !!")
    while True:
        print("\nMenu Principal:")
        print("1. Étude Univariée")
        print("2. Étude Bivariée")
        print("3. Étude Multivariée")
        print("4. Régression Linéaire")
        print("5. Problématique")
        print("6. Quitter")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
            menu_etude_univariee()
        elif choix == "2":
            menu_etude_bivariee()
        elif choix == "3":
            menu_etude_multivariee()
        elif choix == "4":
            menu_regression_lineaire()
        elif choix == "5":
            problematique()
        elif choix == "6":
            print(f"En espérant que cette étude vous ait plu, au revoir {nom_utilisateur}!")
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")


def menu_regression_lineaire():
    while True:
        print("\nMenu Régression Linéaire:")
        print("1. Régression Linéaire de 2 variables")
        print("2. Régression Linéaire de 3 variables")
        print("3. Retour au Menu Principal")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
            regression_linéaire_2variables()
            
        elif choix == "2":
           
            regression_3_variables()
        elif choix == "3":
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")
         






def menu_etude_bivariee():
    while True:
        print("\nMenu Étude Bivariée:")
        print("1. Bivarié Qualitatif vs. Qualitatif")
        print("2. Bivarié Qualitatif vs. Quantitatif")
        print("3. Retour au Menu Principal")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
            bivarie_qualitatif_vs_qualitatif()
        elif choix == "2":
            bivarie_qualitatif_vs_quantitatif()
        
        elif choix == "3":
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")



def menu_etude_multivariee():
    while True:
        print("\nMenu Étude Multivariée:")
        print("1. Nuage de points en 3D")
        print("2. Matrice de Diagrammes en Nuage de Points")
        print("3. Matrice de Covariance")
        print("4. Diagramme en surface 3D")
        print("5. Diagramme dggkhe Contour 3D")
        print("6. Matrice de Corrélation")
        print("7. Retour au Menu Principal")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
            nuage_points_3d()
        elif choix == "2":
            matrice_diagrammes()
        elif choix == "3":
            matrice_covariance()
        elif choix == "4":
            diagramme_surface_3d()
        elif choix == "5":
            diagramme_contour_3d()
        elif choix == "6":
            matrice_correlation()
        elif choix == "7":
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")
    

def nuage_points_3d():
    chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    donnees = pd.read_csv(chemin_fichier)

    donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
    donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
    donnees['Metascore'] = pd.to_numeric(donnees['Metascore'], errors='coerce')

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


def matrice_diagrammes():
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


def matrice_covariance():
    chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    donnees = pd.read_csv(chemin_fichier)

    donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
    donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
    donnees['Runtime'] = donnees['Runtime'].replace(' min', '', regex=True).astype(float)

    colonnes_quantitatives = ['Rating', 'Votes', 'Gross', 'Metascore', 'Runtime']

    matrice_covariance = donnees[colonnes_quantitatives].cov()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_covariance, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.01)
    plt.title('Matrice de Covariance')
    plt.show()


def diagramme_surface_3d():
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


def diagramme_contour_3d():
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
    contour = ax.contour3D(X, Y, Z, 50, cmap='gnuplot')

    fig.colorbar(contour)

    ax.set_xlabel('Votes')
    ax.set_ylabel('Rating')
    ax.set_zlabel('Gross')

    plt.title('Diagramme de Contour 3D')
    plt.show()


def matrice_correlation():
    chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    donnees = pd.read_csv(chemin_fichier)

    donnees['Votes'] = donnees['Votes'].replace(',', '', regex=True).astype(float)
    donnees['Gross'] = donnees['Gross'].replace('[\$,M]', '', regex=True).astype(float) / 1e6
    donnees['Runtime'] = donnees['Runtime'].replace(' min', '', regex=True).astype(float)

    colonnes_quantitatives = ['Rating', 'Votes', 'Gross', 'Metascore', 'Runtime']

    matrice_correlation = donnees[colonnes_quantitatives].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.01)
    plt.title('Matrice de Corrélation')
    plt.show()




def problematique():
    print("\nFonctionnalité Problématique")
    print("Sélectionnez une problématique à étudier :")
    print("1. Il existe une différence significative dans les notes moyennes entre les différents genres de films d'animation.")
    print("2. Il existe une corrélation significative entre l'année de sortie d'un film, ses recettes au box-office et ses notes.")
    print("3. Il existe une différence significative dans les notes moyennes attribuées par les spectateurs pour les films réalisés par différents réalisateurs.")
    print("4. Il existe une corrélation significative entre le certificat d'âge d'un film et ses recettes au box-office.")
    print("5. Il existe une corrélation significative entre la durée d'un film et sa note moyenne.")
    choix = input("Entrez le numéro de la problématique que vous souhaitez étudier : ")

    if choix == "1":
         h1()
    elif choix == "2":
         h2()
    elif choix == "3":
         h3()
    elif choix == "4":
         h4()
    elif choix == "5":
         h5()
    else:
        print("Option invalide. Veuillez sélectionner une option valide.")
        
        
       
        


def menu_etude_univariee():
    while True:
        print("\nMenu Étude Univariée:")
        print("1. Variable Quantitative")
        print("2. Étude Qualitative")
        print("3. Retour au Menu Principal")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
           menu_variable_quantitative()
        elif choix == "2":
           menu_variable_qualitatif()
        elif choix == "3":
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")

def  menu_variable_quantitative():
    while True:
        print("\nMenu Variable Quantitative:")
        print("1. Notes")
        print("2. Nombre de Votes")
        print("3. Budget (Gross)")
        print("4. Metascore")
        print("5. Description")
        print("6. Durée (Runtime)")
        print("7. Retour au Menu Étude Univariée")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
            etude_notes()
        elif choix == "2":
            etude_nombre_votes()
        elif choix == "3":
            etude_budget()
        elif choix == "4":
            etude_metascore()
        elif choix == "5":
            etude_description()
        elif choix == "6":
            etude_duree()
        elif choix == "7":
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")
            
            
def  menu_variable_qualitatif():
    while True:
        print("\nMenu Variable Qualitative:")
        print("1. Certification")
        print("2. Retour au Menu Étude Univariée")
        choix = input("Entrez le numéro de l'option que vous souhaitez sélectionner : ")

        if choix == "1":
            etude_notes()
        elif choix == "2":
            break
        else:
            print("Option invalide. Veuillez sélectionner une option valide.")

def etude_certificat():
    print("Etude de la variable Certification")
    chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
    movies_df = pd.read_csv(chemin_fichier)


    certificate_counts = movies_df['Certificate'].value_counts()


    certificate_table = pd.DataFrame({
    'Certificate': certificate_counts.index,
    'Count': certificate_counts.values
})


    certificate_table['Percentage'] = (certificate_table['Count'] / certificate_table['Count'].sum()) * 100

# Création des plots
    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    sns.barplot(x='Certificate', y='Count', data=certificate_table, palette='viridis')
    plt.title('Nombre de films par Certificat')
    plt.xlabel('Certificat')
    plt.ylabel('Nombre de films')
    plt.xticks(rotation=45, ha='right')


    plt.subplot(1, 2, 2)
    plt.axis('off')  
    plt.table(cellText=certificate_table.values,
          colLabels=certificate_table.columns,
          cellLoc='center',
          colLoc='center',
          loc='center')

plt.suptitle('Analyse univariée qualitative des Certificats des Films', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


      
def etude_notes():
        print("Fonctionnalité Étude des Notes")
           
        chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
        movies_df = pd.read_csv(chemin_fichier)

   
        mean_rating = movies_df['Rating'].mean()
        
        median_rating = movies_df['Rating'].median()
        mode_rating = movies_df['Rating'].mode()[0]
        std_rating = movies_df['Rating'].std()
        var_rating = movies_df['Rating'].var()
        min_rating = movies_df['Rating'].min()
        max_rating = movies_df['Rating'].max()
        quantile_25 = movies_df['Rating'].quantile(0.25)
        quantile_50 = movies_df['Rating'].quantile(0.5)
        quantile_75 = movies_df['Rating'].quantile(0.75)

     
        summary_table = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Écart-type', 'Variance', 'Minimum', 'Maximum', 'Premier quartile (Q1)', 'Deuxième quartile (Q2)', 'Troisième quartile (Q3)'],
            'Valeur': [mean_rating, median_rating, mode_rating, std_rating, var_rating, min_rating, max_rating, quantile_25, quantile_50, quantile_75]
        })

       
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

      
        sns.histplot(movies_df['Rating'], bins=20, kde=True, color='skyblue', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution des Ratings')
        axs[0, 0].set_xlabel('Rating')
        axs[0, 0].set_ylabel('Nombre de films')

  
        sns.boxplot(x=movies_df['Rating'], color='lightblue', ax=axs[0, 1])
        axs[0, 1].set_title('Diagramme en boîte des Ratings')
        axs[0, 1].set_xlabel('Rating')

        sns.kdeplot(movies_df['Rating'], shade=True, color='skyblue', ax=axs[1, 0])
        axs[1, 0].set_title('Courbe de densité des Ratings')
        axs[1, 0].set_xlabel('Rating')
        axs[1, 0].set_ylabel('Densité')

        sns.scatterplot(x=movies_df.index, y=movies_df['Rating'], color='skyblue', ax=axs[1, 1])
        axs[1, 1].set_title('Diagramme de dispersion des Ratings')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Rating')


        cell_text = []
        for row in summary_table.iterrows():
            cell_text.append(row[1].values)

        axs[1, 1].table(cellText=cell_text,
                        colLabels=summary_table.columns,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[1.1, -0.5, 0.5, 0.5],
                        edges='open')

        plt.tight_layout()
        plt.show()

def etude_nombre_votes():
        print("Fonctionnalité Étude du Nombre de Votes")
            

        chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
        movies_df = pd.read_csv(chemin_fichier)

 
        movies_df['Votes'] = movies_df['Votes'].str.replace('.', '').str.replace(',', '.')

        movies_df['Votes'] = movies_df['Votes'].replace('', pd.NA)

        movies_df['Votes'] = pd.to_numeric(movies_df['Votes'], errors='coerce')

        mean_votes = movies_df['Votes'].mean()
        median_votes = movies_df['Votes'].median()
        mode_votes = movies_df['Votes'].mode()[0]
        std_votes = movies_df['Votes'].std()
        var_votes = movies_df['Votes'].var()
        min_votes = movies_df['Votes'].min()
        max_votes = movies_df['Votes'].max()
        quantile_25 = movies_df['Votes'].quantile(0.25)
        quantile_50 = movies_df['Votes'].quantile(0.5)
        quantile_75 = movies_df['Votes'].quantile(0.75)

        
        summary_table = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Écart-type', 'Variance', 'Minimum', 'Maximum', 'Premier quartile (Q1)', 'Deuxième quartile (Q2)', 'Troisième quartile (Q3)'],
            'Valeur': [mean_votes, median_votes, mode_votes, std_votes, var_votes, min_votes, max_votes, quantile_25, quantile_50, quantile_75]
        })

      
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        sns.histplot(movies_df['Votes'], bins=20, kde=True, color='skyblue', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution des Votes')
        axs[0, 0].set_xlabel('Votes')
        axs[0, 0].set_ylabel('Nombre de films')

        sns.boxplot(x=movies_df['Votes'], color='lightblue', ax=axs[0, 1])
        axs[0, 1].set_title('Diagramme en boîte des Votes')
        axs[0, 1].set_xlabel('Votes')

  
        sns.kdeplot(movies_df['Votes'], shade=True, color='skyblue', ax=axs[1, 0])
        axs[1, 0].set_title('Courbe de densité des Votes')
        axs[1, 0].set_xlabel('Votes')
        axs[1, 0].set_ylabel('Densité')


        sns.scatterplot(x=movies_df.index, y=movies_df['Votes'], color='skyblue', ax=axs[1, 1])
        axs[1, 1].set_title('Diagramme de dispersion des Votes')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Votes')

      
        cell_text = []
        for row in summary_table.iterrows():
            cell_text.append(row[1].values)

        axs[1, 1].table(cellText=cell_text,
                        colLabels=summary_table.columns,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[1.1, -0.5, 0.5, 0.5],
                        edges='open')

        plt.tight_layout()
        plt.show()

def etude_budget():
        print("Fonctionnalité Étude du Budget (Gross)")



        chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
        movies_df = pd.read_csv(chemin_fichier)

        
        movies_df['Gross'] = movies_df['Gross'].replace({'\$': '', 'M': ''}, regex=True).astype(float)

       
        mean_gross = movies_df['Gross'].mean()
        median_gross = movies_df['Gross'].median()
        mode_gross = movies_df['Gross'].mode()[0]
        std_gross = movies_df['Gross'].std()
        var_gross = movies_df['Gross'].var()
        min_gross = movies_df['Gross'].min()
        max_gross = movies_df['Gross'].max()
        quantile_25 = movies_df['Gross'].quantile(0.25)
        quantile_50 = movies_df['Gross'].quantile(0.5)
        quantile_75 = movies_df['Gross'].quantile(0.75)


        summary_table = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Écart-type', 'Variance', 'Minimum', 'Maximum', 'Premier quartile (Q1)', 'Deuxième quartile (Q2)', 'Troisième quartile (Q3)'],
            'Valeur': [mean_gross, median_gross, mode_gross, std_gross, var_gross, min_gross, max_gross, quantile_25, quantile_50, quantile_75]
        })

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        sns.histplot(movies_df['Gross'], bins=20, kde=True, color='skyblue', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution des Recettes Brutes')
        axs[0, 0].set_xlabel('Recette Brute ($M)')
        axs[0, 0].set_ylabel('Nombre de films')


        sns.boxplot(x=movies_df['Gross'], color='lightblue', ax=axs[0, 1])
        axs[0, 1].set_title('Diagramme en boîte des Recettes Brutes')
        axs[0, 1].set_xlabel('Recette Brute ($M)')


        sns.kdeplot(movies_df['Gross'], shade=True, color='skyblue', ax=axs[1, 0])
        axs[1, 0].set_title('Courbe de densité des Recettes Brutes')
        axs[1, 0].set_xlabel('Recette Brute ($M)')
        axs[1, 0].set_ylabel('Densité')

        sns.scatterplot(x=movies_df.index, y=movies_df['Gross'], color='skyblue', ax=axs[1, 1])
        axs[1, 1].set_title('Diagramme de dispersion des Recettes Brutes')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Recette Brute ($M)')

      
        cell_text = []
        for row in summary_table.iterrows():
            cell_text.append(row[1].values)

        axs[1, 1].table(cellText=cell_text,
                        colLabels=summary_table.columns,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[1.1, -0.5, 0.5, 0.5],
                        edges='open')

        plt.tight_layout()
        plt.show()


def etude_metascore():
        print("Fonctionnalité Étude du Metascore")


        chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
        movies_df = pd.read_csv(chemin_fichier)

        movies_df['Metascore'] = pd.to_numeric(movies_df['Metascore'], errors='coerce')

        movies_df.dropna(subset=['Metascore'], inplace=True)


        mean_metascore = movies_df['Metascore'].mean()
        median_metascore = movies_df['Metascore'].median()
        mode_metascore = movies_df['Metascore'].mode()[0]
        std_metascore = movies_df['Metascore'].std()
        var_metascore = movies_df['Metascore'].var()
        min_metascore = movies_df['Metascore'].min()
        max_metascore = movies_df['Metascore'].max()
        quantile_25 = movies_df['Metascore'].quantile(0.25)
        quantile_50 = movies_df['Metascore'].quantile(0.5)
        quantile_75 = movies_df['Metascore'].quantile(0.75)

        summary_table = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Écart-type', 'Variance', 'Minimum', 'Maximum', 'Premier quartile (Q1)', 'Deuxième quartile (Q2)', 'Troisième quartile (Q3)'],
            'Valeur': [mean_metascore, median_metascore, mode_metascore, std_metascore, var_metascore, min_metascore, max_metascore, quantile_25, quantile_50, quantile_75]
        })


        fig, axs = plt.subplots(2, 2, figsize=(14, 10))


        sns.histplot(movies_df['Metascore'], bins=20, kde=True, color='skyblue', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution des Metascores')
        axs[0, 0].set_xlabel('Metascore')
        axs[0, 0].set_ylabel('Nombre de films')

        sns.boxplot(x=movies_df['Metascore'], color='lightblue', ax=axs[0, 1])
        axs[0, 1].set_title('Diagramme en boîte des Metascores')
        axs[0, 1].set_xlabel('Metascore')

        sns.kdeplot(movies_df['Metascore'], shade=True, color='skyblue', ax=axs[1, 0])
        axs[1, 0].set_title('Courbe de densité des Metascores')
        axs[1, 0].set_xlabel('Metascore')
        axs[1, 0].set_ylabel('Densité')


        sns.scatterplot(x=movies_df.index, y=movies_df['Metascore'], color='skyblue', ax=axs[1, 1])
        axs[1, 1].set_title('Diagramme de dispersion des Metascores')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Metascore')

        cell_text = []
        for row in summary_table.iterrows():
            cell_text.append(row[1].values)

        axs[1, 1].table(cellText=cell_text,
                        colLabels=summary_table.columns,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[1.1, -0.5, 0.5, 0.5],
                        edges='open')

        plt.tight_layout()
        plt.show()


def etude_description():
        print("Fonctionnalité Étude de la Description")
    
        chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
        movies_df = pd.read_csv(chemin_fichier)

    
        movies_df['Year'] = pd.to_numeric(movies_df['Year'], errors='coerce')
       
        movies_df.dropna(subset=['Year'], inplace=True)


        mean_year = movies_df['Year'].mean()
        median_year = movies_df['Year'].median()
        mode_year = movies_df['Year'].mode()[0]
        std_year = movies_df['Year'].std()
        var_year = movies_df['Year'].var()
        min_year = movies_df['Year'].min()
        max_year = movies_df['Year'].max()
        quantile_25 = movies_df['Year'].quantile(0.25)
        quantile_50 = movies_df['Year'].quantile(0.5)
        quantile_75 = movies_df['Year'].quantile(0.75)

  
        summary_table = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Écart-type', 'Variance', 'Minimum', 'Maximum', 'Premier quartile (Q1)', 'Deuxième quartile (Q2)', 'Troisième quartile (Q3)'],
            'Valeur': [mean_year, median_year, mode_year, std_year, var_year, min_year, max_year, quantile_25, quantile_50, quantile_75]
        })

       
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
        sns.histplot(movies_df['Year'], bins=20, kde=True, color='skyblue', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution des Années de Sortie')
        axs[0, 0].set_xlabel('Année')
        axs[0, 0].set_ylabel('Nombre de films')

        sns.boxplot(x=movies_df['Year'], color='lightblue', ax=axs[0, 1])
        axs[0, 1].set_title('Diagramme en boîte des Années de Sortie')
        axs[0, 1].set_xlabel('Année')

  
        sns.kdeplot(movies_df['Year'], shade=True, color='skyblue', ax=axs[1, 0])
        axs[1, 0].set_title('Courbe de densité des Années de Sortie')
        axs[1, 0].set_xlabel('Année')
        axs[1, 0].set_ylabel('Densité')

        sns.scatterplot(x=movies_df.index, y=movies_df['Year'], color='skyblue', ax=axs[1, 1])
        axs[1, 1].set_title('Diagramme de dispersion des Années de Sortie')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Année')

        
        cell_text = []
        for row in summary_table.iterrows():
            cell_text.append(row[1].values)

        axs[1, 1].table(cellText=cell_text,
                        colLabels=summary_table.columns,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[1.1, -0.5, 0.5, 0.5],
                        edges='open')

        plt.tight_layout()
        plt.show()


def etude_duree():
        print("Fonctionnalité Étude de la Durée (Runtime)")
      
        chemin_fichier = r"C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
        movies_df = pd.read_csv(chemin_fichier)
        movies_df['Runtime'] = movies_df['Runtime'].str.replace(' min', '').astype(int)

        
        mean_runtime = movies_df['Runtime'].mean()
        median_runtime = movies_df['Runtime'].median()
        mode_runtime = movies_df['Runtime'].mode()[0]
        std_runtime = movies_df['Runtime'].std()
        var_runtime = movies_df['Runtime'].var()
        min_runtime = movies_df['Runtime'].min()
        max_runtime = movies_df['Runtime'].max()
        quantile_25 = movies_df['Runtime'].quantile(0.25)
        quantile_50 = movies_df['Runtime'].quantile(0.5)
        quantile_75 = movies_df['Runtime'].quantile(0.75)

        summary_table = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Écart-type', 'Variance', 'Minimum', 'Maximum', 'Premier quartile (Q1)', 'Deuxième quartile (Q2)', 'Troisième quartile (Q3)'],
            'Valeur (min)': [mean_runtime, median_runtime, mode_runtime, std_runtime, var_runtime, min_runtime, max_runtime, quantile_25, quantile_50, quantile_75]
        })

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

  
        sns.histplot(movies_df['Runtime'], bins=20, kde=True, color='skyblue', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution de la durée des films')
        axs[0, 0].set_xlabel('Durée (min)')
        axs[0, 0].set_ylabel('Nombre de films')

  
        sns.boxplot(x=movies_df['Runtime'], color='lightblue', ax=axs[0, 1])
        axs[0, 1].set_title('Diagramme en boîte de la durée des films')
        axs[0, 1].set_xlabel('Durée (min)')

        
        sns.kdeplot(movies_df['Runtime'], shade=True, color='skyblue', ax=axs[1, 0])
        axs[1, 0].set_title('Courbe de densité de la durée des films')
        axs[1, 0].set_xlabel('Durée (min)')
        axs[1, 0].set_ylabel('Densité')

   
        sns.scatterplot(x=movies_df.index, y=movies_df['Runtime'], color='skyblue', ax=axs[1, 1])
        axs[1, 1].set_title('Diagramme de dispersion de la durée des films')
        axs[1, 1].set_xlabel('Index')
        axs[1, 1].set_ylabel('Durée (min)')

   
        cell_text = []
        for row in summary_table.iterrows():
            cell_text.append(row[1].values)

        axs[1, 1].table(cellText=cell_text,
                        colLabels=summary_table.columns,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[1.1, -0.5, 0.5, 0.5],
                        edges='open')

        plt.tight_layout()
        plt.show()

def bivarie_qualitatif_vs_qualitatif():
    print("Analyse Bivariée Qualitatif vs. Qualitatif")
    
    data = pd.read_csv("C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv")

    df = pd.DataFrame({"Genre": data["Genre"], "Director": data["Director"]})

    
    contingency_table = pd.crosstab(df["Genre"], df["Director"])

  
    print("\nTableau de contingence des effectifs :\n", contingency_table)

    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
    print("\nTest du chi2 pour l'indépendance entre Genre et Director :")
    print("Chi2 :", chi2)
    print("p-value :", p)

   
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
    plt.title("Tableau de contingence des effectifs")
    plt.xlabel("Director")
    plt.ylabel("Genre")
    plt.show()

def bivarie_qualitatif_vs_quantitatif():
    print("Analyse des 10 meilleurs directeurs selon le Metascore")
   
    data = pd.read_csv("C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv")

  
    director_scores = data.groupby('Director')['Metascore'].mean().sort_values(ascending=False)

    top_directors = director_scores.head(10)

    # Créer un bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_directors.values, y=top_directors.index, palette='viridis')
    plt.title('Top 10 des directeurs avec la meilleure moyenne de Metascore')
    plt.xlabel('Moyenne de Metascore')
    plt.ylabel('Directeur')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def regression_linéaire_2variables():
    print("Regression linéaire à deux variables")
    chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    donnees = pd.read_csv(chemin_fichier)


    donnees['Votes'] = donnees['Votes'].str.replace(',', '').str.replace('.', '').astype(float)


    donnees['Rating'] = donnees['Rating'].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)


    imputer = SimpleImputer(strategy='mean')
    donnees['Rating'] = imputer.fit_transform(donnees['Rating'].values.reshape(-1, 1))


    X = donnees['Rating'].values.reshape(-1,1) 
    y = donnees['Votes']     


    regression = LinearRegression()


    regression.fit(X, y)


    coef_a = regression.coef_[0]
    coef_b = regression.intercept_


    print("Coefficient a (pente) :", coef_a)
    print("Coefficient b (ordonnée à l'origine) :", coef_b)
    print(donnees['Rating'].unique())


    plt.scatter(X, y, color='blue', label='Données')


    plt.plot(X, regression.predict(X), color='red', label='Régression linéaire')


    plt.xlabel('Rating')
    plt.ylabel('Votes')
    plt.title('Régression linéaire : Rating vs Votes')
    plt.legend()


    plt.show()
def regression_3_variables():
    print("Regression linéaire à 3 variables")
    chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    donnees = pd.read_csv(chemin_fichier)


    donnees = donnees.dropna()


    if donnees['Metascore'].dtype != 'float64':
     donnees['Metascore'] = donnees['Metascore'].str.replace(',', '').str.replace('.', '').astype(float)

    donnees['Gross'] = donnees['Gross'].replace({'\$': '', 'M': ''}, regex=True).astype(float)


    X = donnees[['Rating', 'Year', 'Metascore']]
    y = donnees['Gross']


    regression = LinearRegression()
    regression.fit(X, y)


    print("Coefficients a (pentes) :", regression.coef_)
    print("Coefficient b (ordonnée à l'origine) :", regression.intercept_)


    x_surf, y_surf = np.meshgrid(np.linspace(donnees['Rating'].min(), donnees['Rating'].max(), 100),
                              np.linspace(donnees['Year'].min(), donnees['Year'].max(), 100))
    X_surf = pd.DataFrame({'Rating': x_surf.ravel(), 'Year': y_surf.ravel(), 'Metascore': np.repeat(donnees['Metascore'].mean(), 10000)})


    predicted = regression.predict(X_surf)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(donnees['Rating'], donnees['Year'], donnees['Metascore'], c='blue', marker='o', alpha=0.5)
    ax.plot_surface(x_surf, y_surf, predicted.reshape(x_surf.shape), color='None', alpha=0.3)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Year')
    ax.set_zlabel('Metascore')
    plt.show()
def h1():
    print('Etude Hypothèse 1')
    file_path = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    df = pd.read_csv(file_path)


    df.dropna(inplace=True)

    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')


    genre_rating = df.groupby('Genre')['Rating'].mean().sort_values(ascending=False)


    print("Moyenne des Ratings pour chaque genre de film:")
    print(genre_rating)


    adventure_group = df[df['Genre'].str.contains('Adventure', case=False)]
    non_adventure_group = df[~df['Genre'].str.contains('Adventure', case=False)]
    rating_adventure = adventure_group['Rating']
    rating_non_adventure = non_adventure_group['Rating']
    
    stat, p_value = mannwhitneyu(rating_adventure, rating_non_adventure)
    print("Statistique de test de Mann-Whitney U:", stat)
    print("p-valeur:", p_value)

    t_stat, p_value = ttest_ind(adventure_group['Rating'], non_adventure_group['Rating'])
    print("t-statistique:", t_stat)
    print("p-valeur:", p_value)


    X = df[['Metascore']]
    y = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    print("Coefficient de régression:", regression_model.coef_[0])
    print("Coefficient d'interception:", regression_model.intercept_)

    y_pred = regression_model.predict(X_test)
    accuracy = accuracy_score(y_test.round(), y_pred.round())
    print("Précision de la régression linéaire:", accuracy)
#Anova : la p-valeur est de 0.062, légèrement supérieure au seuil traditionnel de 0.05.
#Le coefficient de régression est d'environ 0.014.
# Cela signifie que pour chaque unité d'augmentation dans le score de Metascore, le score de Rating augmente d'environ 0.014 en moyenne.
#ecart entre les moyennes de genre, t-stat

def h3():
    print("Etude Hypothèse 3")
    file_path = "C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv"
    df = pd.read_csv(file_path)


    df.dropna(subset=['Director', 'Rating', 'Metascore'], inplace=True)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')


    directors_ratings = df.groupby('Director')['Rating'].agg(['mean', 'count', 'std'])
    min_films = 3  
    filtered_directors = directors_ratings[directors_ratings['count'] >= min_films].index


    anova_data = {director: df[df['Director'] == director]['Rating'] for director in filtered_directors}


    if len(anova_data) > 1:
        f_stat, p_value = f_oneway(*anova_data.values())
        print("F-statistique ANOVA:", f_stat)
        print("p-valeur ANOVA:", p_value)

    for director in filtered_directors:
         director_data = df[df['Director'] == director]
         X = director_data[['Metascore']]
         y = director_data['Rating']
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)

  
    print(f"Résultats pour {director}:")
    print("Coefficient de régression (pente):", regression_model.coef_[0])
    print("Coefficient d'interception:", regression_model.intercept_)
    mse = mean_squared_error(y_test, y_pred)
    print("Erreur quadratique moyenne:", mse)
    print("-" * 50)


    if len(filtered_directors) > 1:
      ratings1 = df[df['Director'] == filtered_directors[0]]['Rating']
      ratings2 = df[df['Director'] == filtered_directors[1]]['Rating']
      t_stat, p_value = ttest_ind(ratings1, ratings2)
      print(f"T-test entre {filtered_directors[0]} et {filtered_directors[1]}:")
      print("T-statistique:", t_stat)
      print("p-valeur:", p_value)
      print ("L'ANOVA et le T-test ne montrent pas de différences significatives entre leurs notes moyennes de films.")
      print ("Cela indique que, dans l'ensemble, les réalisateurs n'ont pas un impact significatif sur les notes moyennes des films, du moins dans le cadre de cette analyse.")




def h2():
    print("Etude statistique de la 2ème hypothèse")



    file_path = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    df = pd.read_csv(file_path)


    df.dropna(subset=['Year', 'Gross'], inplace=True)


    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Gross'] = df['Gross'].str.replace('$', '').str.replace('M', '').astype(float)

    X = df[['Year']]
    y = df['Gross']

    regression_model = LinearRegression()


    regression_model.fit(X, y)


    coef = regression_model.coef_[0]
    intercept = regression_model.intercept_

    print("Coefficient de régression:", coef)
    print("Coefficient d'interception:", intercept)


    plt.scatter(X, y, color='blue')
    plt.plot(X, regression_model.predict(X), color='red')
    plt.title('Régression linéaire entre l\'année de sortie et les recettes au box-office')
    plt.xlabel('Année de sortie')
    plt.ylabel('Recettes au box-office ($M)')
    plt.show()
    print("Ainsi, en réponse à la problématique initiale, il semble y avoir une relation positive entre la note critique d'un film et sa performance en termes de notes attribuées par les utilisateurs. Cependant, d'autres facteurs peuvent également influencer les recettes au box-office, et une analyse plus approfondie serait nécessaire pour les examiner.")
    
    
    
    
    
    
def h5():
    print("Etude statistique de la 4ème hypothèse")
    file_path = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    df = pd.read_csv(file_path)


    df.dropna(inplace=True)


    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Runtime'] = pd.to_numeric(df['Runtime'].str.replace(" min", ""), errors='coerce')


    X = df[['Runtime']]
    y = df['Rating']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    regression_model = LinearRegression()


    regression_model.fit(X_train, y_train)


    print("Coefficient de régression:", regression_model.coef_[0])
    print("Coefficient d'interception:", regression_model.intercept_)

    y_pred = regression_model.predict(X_test)

 
    accuracy = regression_model.score(X_test, y_test)
    print("Précision de la régression linéaire:", accuracy)
    print("En conclusion, bien qu'il y ait une tendance légèrement positive entre la durée d'un film et sa note moyenne selon les données, il est important de noter que la précision du modèle est faible")
    print("Cela suggère que d'autres facteurs non pris en compte dans cette analyse peuvent influencer la note moyenne des films. Par conséquent, la durée seule ne peut pas être considérée comme un prédicteur fiable de la note moyenne d'un film.")
    
def h4():
    print("Etude statistique de la 5 èeme hypothèse")
    file_path = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
    df = pd.read_csv(file_path)


    df.dropna(inplace=True)


    df['Gross'] = df['Gross'].replace({'\$': '', 'M': 'e6'}, regex=True).astype(float)


    correlation, p_value = pearsonr(df['Year'], df['Gross'])
    print("Corrélation entre l'année de sortie et les recettes au box-office:", correlation)
    print("p-valeur:", p_value)


 




menu_principal()