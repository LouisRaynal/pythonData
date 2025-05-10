# -*- coding: utf-8 -*-
"""
Correction de l'examen d'Analyse de données avec Python

@author: Louis RAYNAL
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

pd.options.display.max_columns = None

# Question 1

# Question 2 (0.5pt total)
## chargement (0.25pt)
os.chdir('/Users/Guest/Desktop')
#os.chdir('D:/Louis/Documents/Cours_AnalyseDeDonneesPython/Examen1_2024_2025')
os.getcwd()
os.listdir()
df = pd.read_csv('maisons.csv', header=0, skipfooter=0, sep=',')
## Vérification (0.25pt)
df.shape

df.columns

# Question 3 (1pt total)
# Changement (0.75pt)
dicoNoms = {'id':'idMaison', 'year':'anneeVente',
            'price':'prix', 'bedrooms':'nbChambres',
            'bathrooms':'nbSdB', 'sqft_living':'sqftHabitable',
            'sqft_lot':'sqftTerrain', 'floors':'nbEtages',
            'waterfront':'vueMer', 'view':'vueNote', 
            'condition':'conditionNote', 'grade':'categorieNote',
            'sqft_above':'sqftHaut', 'sqft_basement':'sqftBas',
            'yr_built':'anneeConstr', 'yr_renovated':'anneeRenov'}
df.rename(columns=dicoNoms, inplace=True)
# Vérification (0.25pt)
df.columns

# Question 4 (1pt total)
# Présence de doublons ? 
df.duplicated().sum() # il y a 2 lignes doublons : 0.25pt
# On supprime les lignes dédoublonnées
df.drop_duplicates(inplace=True) # 0.5pt
# On vérifie qu'il n'y a bien plus de doublons
df.duplicated().sum() # 0.25pt

# Question 5 (1pt total)
# (a) 0.25
df.isnull().sum(axis=0) # 2 dans les colonnes nbChambres et anneeRenov
# (b) 0.5
dicoRemplacement = {'nbChambres':df.nbChambres.median(), 'anneeRenov':0}
df.fillna(dicoRemplacement, inplace=True)
# (c) 0.25
df.isnull().sum(axis=0)

# Question 6 (1pt total)
# (a) 0.25
df.loc[df.idMaison == 123456,:] # année de vente : 1015
# (b) 0.5
df.loc[df.idMaison == 123456,'anneeVente'] = 2015
# (c) 0.25
df.loc[df.idMaison == 123456,:]

# Question 7 (1pt total)
( df.sqftHabitable != df.sqftHaut + df.sqftBas ).sum()
# Il y a bien 0 incohérences

# Question 8 (0.5pt total)
df.anneeVente.value_counts() # 6980 maisons vendues en 2015

# Question 9 (0.5pt total)
df.anneeVente.value_counts(normalize=True) # 67.7% des ventes ont eu lieu en 2014

# Question 10 (1pt total)
tabvueMerAnneeEff = pd.crosstab(df.vueMer, df.anneeVente, margins=True)
tabvueMerAnneeEff # 114 maisons avec vue sur la mer vendues en 2014

# Question 11 (1pt total : juste pour le calcul)
distChambresParEtages = pd.crosstab(df.nbChambres, df.nbEtages, normalize='columns', margins=True)
distChambresParEtages

# Traçons ces distributions pour faciliter cette comparaison (optionnel)
distChambresParEtages.plot.bar()
# Le nombre d'étages ne semble pas beaucoup affecter le nombre de chambres.
# En effet, les distributions se ressemblent beaucoup par nombre d'étages ou tous étages confondus.
# On peut donc penser que ces deux variables sont indépendantes.
# Ce commentaire est subjectif, il est difficile de dire à l'oeil nu s'il y a dépendance ou non.

# Question 12 (1pt total)
tabEff = pd.crosstab(df.nbChambres, df.nbEtages)
tabEff

stat_chi2, p_value, dof, expected = ss.chi2_contingency(tabEff)
print(stat_chi2) # La statistique du chi-2 est très éloignée de 0
print(p_value) # La p-value est aussi inférieure à 5%
# On rejette donc l'hypothèse d'indépendance, les deux variables sont dépendantes.

# Question 13 (1pt total : 0.5 pour les stats descriptives, 0.5 pour l'histogramme et commentaires)
df.prix.min()
df.prix.max()
df.prix.mean()
df.prix.median()
df.prix.plot.hist(bins=100, density=True)
# On observe que la majorité des données sont concentées autour de 75000 et 1000000,
# avec un pic autour de la moyenne/médiane.
# Le bien à 7700000 étend l'axe des abscisses jusqu'à 8 millions.

# Question 14  (1.5pt total)
# (0.5pt)
q1 = df.prix.quantile(q=0.25)
q1
q3 = df.prix.quantile(q=0.75)
q3
# Intervalle : [ 321725 ; 645000 ]
# Une possibilité : (1pt)
( (df.prix <= q3) & (df.prix >=q1) ).mean()
# Une autre possibilité :
df.loc[ (df.prix <= q3) & (df.prix >=q1) ].shape[0] / df.shape[0]

# Mettre 0.25 si intervalle obtenu par un boxplot

# Question 15 (0.75pt total)
df.loc[df.nbSdB == df.nbSdB.max(), 'prix'] # 7700000 et 2280000 dollars

# Question 16 (0.5pt total)
df.plot.box(column='prix', by='categorieNote')

# Question 17 (1pt total : 0.5 pour calculs hors R2, 0.25 pour calcul du R2, 0.25 pour le commentaire)
varGroup = df.groupby('categorieNote')['prix'].var(ddof=0)
varGroup

countGroup = df.groupby('categorieNote').size()
countGroup

varianceIntra = ( countGroup * varGroup ).sum() / countGroup.sum()

varianceTotale = df['prix'].var(ddof=0)

Rsquare = 1 - varianceIntra/varianceTotale
Rsquare
# Le R2 est de 0.52, on peut donc dire que le grade a un effet important sur le prix
# de vente de la maison.

# Question 18 (1.5pt total : 0.5 pour le sort_value, 0.5 pour l'extraction de la colonne prix, et 0.5 pour l'extraction des 3 valeurs d'intérêt)
df.corr(numeric_only=True) 
df.corr(numeric_only=True).sort_values(by='prix', ascending=False).prix[1:4]
# Pour cette question compter 0.75 si extraction "à l'oeil" (en observant les valeurs de df.corr())
# Pour cette question compter 1 si extraction "à la main" (calcul des correlations à la main)

# Question 19 (0.5pt total : 0.25 pour le graph, 0.25 pour le commentaire)
df.plot.scatter(x='sqftHabitable', y='prix')
# Le prix augmente avec la superficie habitable.
# On observe aussi que les prix sont de plus en plus variables lorsque la superficie augmente.

# Question 20 (1.25pt total)
# (a) (0.25pt)
res = ss.linregress(df.sqftHabitable, df.prix)
a_hat = res.slope
a_hat
b_hat = res.intercept
b_hat

# (b) (0.5pt)
val_prix = a_hat * df.sqftHabitable + b_hat
df.plot.scatter(x='sqftHabitable', y='prix')
plt.plot(df.sqftHabitable, val_prix, c='r') # c='r' pour couleur rouge

# (c) (0.5pt)
# Combien vaut 300 mètres carrés en pieds carrés
300/0.09290304 
a_hat * (300/0.09290304) + b_hat
# environ 862605 dollars

# Question 21 (1pt total)
# (a) (0.5pt)
groupeAnneeConstr = pd.cut(df.anneeConstr, bins=[1895,1915,1935,1955,1975,1995,2015], include_lowest=True)
# (b) (0.5pt)
df.groupby(groupeAnneeConstr)['prix'].agg(func=[np.mean, np.median, np.std])
# Le prix médian des ventes est le plus bas pour les maisons construites entre 1935 et 1975
# alors que les maisons vendues les plus cher sont celles construites avant 1935 ou après 1995


# Question 22 (1.5pt total)
### Créez une nouvelle colonne dans votre DataFrame nommée renov
### qui prend la valeur 'non rénovée' si la maison n'a pas été rénovée, et 'rénovée' sinon
# (0.75pt)
def renovation(x) :
    if x == 0 :
        return 'non rénovée'
    else :
        return 'rénovée'
# (0.5pt)
df['reno'] = df.anneeRenov.map(renovation)
# Vérifiez que vous avez bien uniquement les deux modalités attendues
df.reno.unique()

# (0.5pt : compter ok si cette ligne)
df.groupby('reno')['prix'].mean()
# Comme attendu, les maisons rénovées sont vendues plus cher que les non-rénovées
