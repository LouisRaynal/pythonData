# -*- coding: utf-8 -*-
"""
Correction de l'examen d'Analyse de données avec Python, du 22/04/2024

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
# Chargement (0.25pt)
#os.chdir('/Users/Guest/Desktop')
os.chdir('D:/Louis/Documents/Cours_AnalyseDeDonneesPython/Examen1_2023_2024')
os.getcwd()
os.listdir()
df = pd.read_csv('voitures.csv', header=0, skipfooter=0, sep=',')
## Vérification (0.25pt)
df.shape
# ou
df.columns

# Question 3 (1pt total)
# Changement (0.75pt)
dicoNoms = {'Car_Name':'nom',
            'Year':'anneeAcquis',
            'Selling_Price':'prixVente', 
            'Present_Price':'prixActuel', 
            'Kms_Driven':'km',
            'Fuel_Type':'carburant',
            'Seller_Type':'typeVendeur', 
            'Transmission':'transmission',
            'Owner':'nbProp'}
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
# (a) 0.25pt
df.isnull().sum(axis=0)
# (b) 0.5pt
dicoRemplacement = {'prixVente':df.prixVente.median(),
                    'nbProp':0}
df.fillna(dicoRemplacement, inplace=True)
# (c) 0.25pt
df.isnull().sum(axis=0)


# Question 6 (1pt total)
# (a) 0.25pt
df.loc[df.nom == 'TVS Jupyter',:]
# (b) 0.5pt
df.loc[df.nom == 'TVS Jupyter','typeVendeur'] = 'Individual'
# (c) 0.25pt
df.loc[df.nom == 'TVS Jupyter',:]


# Question 7 (1pt total)
(df.prixVente > df.prixActuel).sum()
# Il y a bien 0 incohérence

# Question 8 (0.5pt total)
df.anneeAcquis.value_counts() # 15 voitures de 2010 vendues

# Question 9 (0.5pt total)
df.anneeAcquis.value_counts(normalize=True) # 20.0669% des ventes concernent des voitures de 2015

# Question 10 (1pt total)
tabAnneeCarbuEff = pd.crosstab(df.anneeAcquis, df.carburant, margins=True)
tabAnneeCarbuEff # 14 voitures Diesel de 2014

# Question 11 (1pt total : 0.75pt pour le calcul + 0.25pt pour l'interprétation justifiée)
distCarbuTrans = pd.crosstab(df.carburant, df.transmission, normalize='columns', margins=True)
distCarbuTrans

# Traçons ces distributions pour faciliter cette comparaison (optionnel)
distCarbuTrans.plot.bar()
# La transmission ne semble pas affecter la distribution du carburant.
# En effet, les distributions se ressemblent beaucoup par type de transmission ou toute transmission.
# On peut donc penser que ces deux variables sont indépendantes.
# Ce commentaire est subjectif, il est difficile de dire à l'oeil nu s'il y a dépendance ou non.

# Question 12 (1pt total)
tabEff = pd.crosstab(df.carburant, df.transmission)
tabEff

stat_chi2, p_value, dof, expected = ss.chi2_contingency(tabEff)
print(stat_chi2) # La statistique du chi-2 est peu éloignée de 0
print(p_value) # La p-value n'est pas inférieure à 5%
# On ne rejette donc pas l'hypothèse d'indépendance, les deux variables sont indépendantes.

# Question 13 (1pt total : 0.5pt pour les stats descriptives, 0.5pt pour l'histogramme et commentaires)
df.prixVente.min()
df.prixVente.max()
df.prixVente.mean()
df.prixVente.median()
df.prixVente.plot.hist(bins=35, density=True)
# On observe qu'une grande partie des voitures sont vendues à un prix entre 0 et 1000 dollars.
# On observe ensuite que la plupart des ventes sont faites à un prix inférieur à 5000 dollars.

# Question 14  (1.5pt total)
# (0.5pt)
q1 = df.prixVente.quantile(q=0.25)
q1
q3 = df.prixVente.quantile(q=0.75)
q3
# Intervalle : [ 0.85 ; 6.0 ]
# Une possibilité : (1pt)
( (df.prixVente <= q3) & (df.prixVente >=q1) ).mean()
# Une autre possibilité :
df.loc[ (df.prixVente <= q3) & (df.prixVente >=q1) ].shape[0] / df.shape[0]


# Question 15 (0.75pt total)
df.loc[df.prixVente == (df.prixActuel-1), 'nom'] # brio et city

# Question 16 (0.5pt total)
df.plot.box(column='prixVente', by='transmission')

# Question 17 (1pt total : 0.5pt pour calculs hors R2, 0.25pt pour calcul du R2, 0.25pt pour le commentaire)
varGroup = df.groupby('transmission')['prixVente'].var(ddof=0)
varGroup

countGroup = df.groupby('transmission').size()
countGroup

varianceIntra = ( countGroup * varGroup ).sum() / countGroup.sum()

varianceTotale = df['prixVente'].var(ddof=0)

Rsquare = 1 - varianceIntra/varianceTotale
Rsquare
# Le R2 est de 0.12, on peut donc dire que le type de transmission a un faible effet sur le prix
# de vente de la voiture.

# Question 18 (2pt total : 0.75pt pour le sort_value, 0.5pt pour l'extraction de la colonne prix, et 0.75pt pour l'extraction des 2 valeurs d'intérêt)
df.corr()
df.corr().sort_values(by='prixVente', ascending=False).prixVente[1:3]
# Pour cette question compter 0.5pt si extraction "à l'oeil" (en observant les valeurs de df.corr())
# Pour cette question compter 1pt si extraction "à la main" (calcul des correlations à la main)

# Question 19 (0.5pt total : 0.25pt pour le nuage et droite, 0.25pt pour le commentaire)
df.plot.scatter(x='prixVente', y='prixActuel')
# Le prix neuf augmente avec le prix d'occasion.
# On observe aussi que les prix sont de plus en plus variables lorsque les prix augmentent.

# Question 20 (1pt total)
# (a) (0.25pt)
res = ss.linregress(df.prixVente, df.prixActuel)
a_hat = res.slope
a_hat
b_hat = res.intercept
b_hat

# (b) (0.25pt)
valPrixActuel = a_hat * df.prixVente + b_hat
df.plot.scatter(x='prixVente', y='prixActuel')
plt.plot(df.prixVente, valPrixActuel, c='r') # c='r' pour couleur rouge

# (c) (0.5pt)
a_hat * (30000/1000) + b_hat
# environ 45822 dollars

# Question 21 (1pt total)
# (a) (0.5pt)
groupeAnneeAcquis = pd.cut(df.anneeAcquis, bins=[2000,2005,2010,2015,2020], include_lowest=False)
# (b) (0.5pt)
df.groupby(groupeAnneeAcquis)['prixVente'].agg(func=[np.mean, np.median, np.std])
# Le prix médian de vente tend à augmenter avec l'année d'acquisition.

# Question 22 (1.25pt total)

# (0.75pt)
def premierPropFunc(x) :
    if x == 0 :
        return 'Oui'
    else :
        return 'Non'
df['premierProp'] = df.nbProp.map(premierPropFunc)

# (0.25pt)
# Vérifiez que vous avez bien uniquement les deux modalités attendues
df.premierProp.unique()

# (0.25pt)
df.groupby('premierProp')['prixVente'].mean()
# Les voitures n'ayant jamais eu d'anciens propriétaires (autre que celui qui met la voiture en vente)
# ont un prix de vente plus élevé