import pandas as pd
from scipy.stats import f_oneway

# Charger les données
chemin_fichier = r'C:/Users/hugop/OneDrive/Bureau/Projet-Jeux de données (1)/Projet-Jeux_de_donne╠ües/Films_Animation.csv'
donnees = pd.read_csv(chemin_fichier)


variable_qualitative = 'Genre' 
variable_continue = 'Rating'


donnees = donnees[[variable_qualitative, variable_continue]].dropna()


resultats_anova = f_oneway(*(donnees[variable_continue][donnees[variable_qualitative] == categorie] for categorie in donnees[variable_qualitative].unique()))

# Afficher les résultats
print("Test ANOVA:")
print(f"Statistic F : {resultats_anova.statistic}")
print(f"P-value : {resultats_anova.pvalue}")
