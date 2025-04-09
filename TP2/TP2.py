# -*- coding: utf-8 -*-
"""
TP 2 d'analyse de données avec Python
"""

# Chargement des librairies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

#------- 1. Premier jeu de données : céréales

### Exercices :

# 1. La population de l'étude est l'ensemble des marques de paquets de céréales vendus aux Etats-Unis,
# 2. 
# Variables qualitatives nominales : produit, fabricant, type
# Variables qualitatives ordinales : étagère
# Variables quantitatives continues : calories, protéines, lipides, sodium, fibres, glucides,
#   sucres, potassium, poids, tasses, note
# Variables quantitatives discrètes : vitamines (car uniquement trois modalités possibles)    

#------- 2. Inspection préliminaire du jeu de données

### Exercices :

# 2. La première ligne contient bien l'entête
# 3. Il n'y a pas de lignes inutiles en fin de fichier
# 4. Les données sont séparées par des virgules
# 5. IL n'y a pas de valeurs manquantes à première vue (des vides dans nos données)
# 6. 79 lignes avec l'entête, et 16 colonnes

#------- 3. Chargement des données

# 1. Se placer dans le répertoire où se trouve le jeu de données (le bureau ici)
os.chdir('/Users/Guest/Desktop')
#os.chdir('D:/Louis/Documents/Cours_AnalyseDeDonneesPython/TP2')
# puis vérifier que l'on est sur le bon répertoire
os.getcwd()

# 2. Lister les fichiers se trouvant dans ce répertoire
os.listdir()

# 3. Lire le fichier de jeu de données dans Python, ici sous la forme d'un DataFrame
fichier = 'cereales.csv'
df = pd.read_csv(fichier, header=0, skipfooter=0, sep=',')

### Exercices :
# 1.
# Vérifier que tout est OK
df
df.shape
# Nous avons bien 78 lignes (hors entête), 16 colonnes

# Aperçu des données
df.head()
df.tail()

# 2.
# header : indique le numéro de ligne où se trouve l'entête dans cereales.csv
# skipfooter : indique combien de ligne à la fin du fichier il faut sauter
# sep : désigne le caractère qui sépare nos données


# Afin d'afficher toutes les colonnes
pd.options.display.max_columns = None

#------- 4. Nettoyage des données

#------- Les doublons

#### Détection de doublons de lignes

df.duplicated() # affiche True sur la ligne qui apparait en double (False pour la première apparition)

# il est alors facile de compter le nombre de doublons
df.duplicated().sum() # 1 ligne doublon

# Quoi qu'il en soit, les 2 dernières lignes du jeu de données posent problèmes
df.drop_duplicates(inplace=True)


### Exercices :
# 1.
df.duplicated(['produit']).sum()
# 2.
df.duplicated(['type']).sum()



#---------- Détection de valeurs manquantes 

df.isnull()

# On compte combien de fois True apparait par colonne
df.isnull().sum(axis=0)

df.isnull().sum(axis=1) # Idem mais avec les lignes


# Ou bien ceci, pour un état général du jeu de données
df.info() 


### Curiosité : les -1 sont des valeurs sentinelles
df == -1 # On compare toutes les valeurs de df avec -1

# On compte combien il y en a par colonne
(df == -1).sum(axis=0)

# On souhaite maintenance visualiser les lignes et colonnes avec ces -1
(df == -1).any(axis=1) # Pour chaque ligne
(df == -1).any(axis=0) # Pour chaque colonne

# On extrait de df (grâce à loc), les lignes et colonnes où True est retourné
# par les deux formules utilisant .any
df.loc[ (df == -1).any(axis=1) , (df == -1).any(axis=0) ]


### Rechargeons le jeu de données en disant que les -1 correspondent à des valeurs manquante
df = pd.read_csv(fichier, header=0, skipfooter=0, sep=',', na_values=[-1])

### Exercices :
# 1. On resupprime les doublons
df.drop_duplicates(inplace=True)

# 2. Il y a maintenant des valeurs manquantes
df.isnull()
df.isnull().sum(axis=0)

# 3. Visualisons les
df.isnull().any(axis=0)
df.isnull().any(axis=1)

df.loc[ df.isnull().any(axis=1) , df.isnull().any(axis=0) ]


### Que faire avec ces valeurs manquantes ?

# Option 1. : Supprimer les lignes correspondantes
df.dropna() # l'absence de inplace=True dans cette ligne ne modifiera pas df

# Option 2. : Remplacement par la moyenne
dico = {'glucides':df.glucides.mean(),
        'sucres':df.sucres.mean(),
        'potassium':df.potassium.mean()}
dico

df.fillna(dico, inplace=True) # inplace=True permet bien de modifier df

df.loc[[4,20,57],['glucides','sucres','potassium']]


#---------- Identification de données aberrantes

df.describe()
#df.describe(include='all')

# Pour sélectionner les colonnes uniquement numériques
df.select_dtypes(include=[np.number])

col_num = df.select_dtypes(include=[np.number]).columns

# Au passage pour les non numériques
df.select_dtypes(include=[object])

# On veut identifier les valeurs qui sont négatives parmis cette table numérique

(df.loc[:,col_num] < 0)

(df.loc[:,col_num] < 0).sum(axis=0) # le nombre de négatifs dans chaque colonne
(df.loc[:,col_num] < 0).sum(axis=1) # les lignes correspondantes

# Voyons voir les lignes et colonnes qui posent problème
df[col_num].loc[ (df.loc[:,col_num] < 0).any(axis=1), (df.loc[:,col_num] < 0).any(axis=0) ]


### Exercices :
# df < 0
# On obtient une erreur car on ne peut pas comparer des colonnes non numériques avec 0


# Pour résoudre ce problème, appliquons sur toutes les colonnes numériques, la fonction abs()
# et on remplace df
df.loc[:,col_num] = df.loc[:,col_num].abs()
df


#---------- Transformation de données


### Recodage d'une colonne

dicoType = {'F':'froides', 'C':'chaudes'}

df.type = df.type.map(dicoType)

# autre manière avec une fonction

def decodeType(x):
     if x == 'F':
         return 'froides'
     elif x == 'C':
         return 'chaudes'
     else:
         return x

df.type = df.type.map(decodeType)

### Exercices :
# Faisons de même pour le fabricant :
dicoFabr = {'A':'American Home Food Products',
           'G':'General Mills',
           'K':'Kelloggs',
           'N':'Nabisco',
           'P':'Post',
           'Q':'Quaker Oats',
           'R':'Ralston Purina'}

df.fabricant = df.fabricant.map(dicoFabr)


### Renommer des lignes ou colonnes

df.rename(columns=str.upper)

df.rename(index={0:'zero'})

df.rename(index={0:'zero',1:'un'}, columns=str.upper)

# Remarque : Les lignes ci-dessus n'ont pas modifier df
# pour cela il faut ajouter inplace=True
df.columns # par exemple les noms de colonnes sont toujours en minuscules

### Exercices :
# On renomme la colonne étagèr en étagère, et on modifie bien df avec inplace=True
df.rename(columns={'étagèr':'étagère'}, inplace=True)

df.columns # le nom de colonne a bien été modifié

### Conversion de colonnes

df.tasses = df.tasses*23.66

df.poids = df.poids*28.35

### Création d'une colonne
df['Produit_maj'] = df.produit.str.upper()
df

### Suppression
df.drop(columns = 'Produit_maj', inplace=True)


# ------- 5. Exploration / Analyse du jeu de données

# ------- Visualisation des données -----------

### Tracer des graphs avec pandas et seaborn

# Des courbes

df.plot.line()

df[['calories', 'potassium']].plot.line()

df[['sucres','calories', 'potassium']].plot.line(subplots=True, sharex=True)

df[['sucres','calories', 'potassium']].plot.line(figsize=(7,7), title='Ceci est mon titre')


df.plot.line(x='sucres', y='calories')

df.sort_values(by='sucres').plot.line(x='sucres', y='calories')


# Diagramme en bâton

df.type.value_counts()
df.type.value_counts().plot.bar()

### Exercices :
df.fabricant.value_counts().plot.bar()

### Bonus :
df.étagère.value_counts().plot.bar()


### Camemberts

df.type.value_counts().plot.pie(ylabel='', autopct='%.1f')

### Exercices :
df.fabricant.value_counts().plot.pie(ylabel='', autopct='%.2f')

### Bonus :
df.étagère.value_counts().plot.pie(autopct='%.2f')


### Histogramme

# bins permet de partitionner en 10 classes
df.calories.plot.hist(bins=10)

df.calories.plot.hist(bins=10, density=True)


### Exercices :
df.note.plot.hist(bins=15, density=True)
# On remarque que la distribution des notes se concentre aux alentours de 30 - 50
# et qu'il y a une barre qui se détache avec une très haute note

# Equivalent seaborn
# sns.distplot(df.calories, bins=10)

### Nuages de points

df.plot.scatter(x='lipides', y='calories')


### Exercices :
# 1.
df.plot.scatter(x='calories', y='note')
# On constate  que lorsque les calories augmentent, la note diminue

# 2.
df.plot.scatter(x='sucres', y='note')
# On constate  que lorsque les sucres augmentent, la note diminue


# seaborn permet de tracer des pairplots

df.columns

sns.pairplot(df[['calories', 'protéines', 'lipides','sodium', 'note']])

sns.pairplot(df[['fibres', 'glucides', 'sucres', 'potassium', 'note']])


### Boxplots

df.plot.box(column='note', rot=90) # rot permet de faire une rotation des libellés

df.plot.box(column='note', by='étagère',rot=90)

df.plot.box(column=['sucres','note'], by='étagère',rot=90)

df.plot.box(column=['fibres','note'], by='étagère',rot=90)

### Exercices :
df.plot.box(column='note', by='fabricant',rot=90)
df.plot.box(column=['fibres','note'], by='fabricant',rot=90)
# Le fabricant Nabisco produit des céréales majoritairement avec haute teneur en fibres
# (car son boxplot est compact est situé aux alentours de 3.5)
# Ce fabricant a aussi les meilleurs notes


#### ---------- 

## Caractéristiques de la distribution centrale d'une variable

# Moyenne
df.sucres.mean()
np.mean(df.sucres)

# Médiane
df.sucres.median()
np.median(df.sucres)

# Mode
df.type.mode()

## Caractéristiques de la variabilitié d'une variable

# Variance
df.sucres.var(ddof=0)
np.var(df.sucres)
# Note : il existe une version de la variance avec N-1 au lieu de N au dénominateur.
# Par défaut pandas utilise N-1, lui préciser ddof=0 permet d'utiliser N.
# Par défaut numpy utilise N

# Ecart-type
df.sucres.std(ddof=0)
np.std(df.sucres)

# Ecart inter-quartile
# Calcul de quantiles (ici les 3 quartiles)
df.sucres.quantile(q=[0.25,0.5,0.75])
# Ecart inter-quartile :
df.sucres.quantile(q=0.75) - df.sucres.quantile(q=0.25)

# Etendue
np.ptp(df.sucres)
# vérification
df.sucres.max() - df.sucres.min()


# Describe
df.select_dtypes(include=[np.number]).describe()

df.select_dtypes(include=[object]).describe()    

# Ou encore plus simple :
df.describe(include=[np.number])

df.describe(include=[object])

#---------------------------------------------------

### Exercices :
    
### Point sur la détection de valeurs aberrantes
# Une règles de détection consiste à identifier les
# points tombant en dehors d'un interval égal à
# [Q1 - 1.5*IQR ; Q3 + 1.5*IQR]
# Ce 1.5 provient du fait que pour une loi normale
# les valeurs en dehors de +/-3 écart type
# sont considérées comme outliers

# On stocke dans une variable chaque élément permettant de calculer les bornes
Q1 = df.note.quantile(q=0.25)
Q3 = df.note.quantile(q=0.75)
IQR = Q3 - Q1

# On calcule les bornes
borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR

# On compare df.note avec borne_basse et borne_haute
condition1 = df.note < borne_basse
condition2 = df.note > borne_haute

# On combine les deux conditions en les séparant par l'opérateur logique "ou" : |
conditionFinale = condition1 | condition2
# les notes qui nous interessent sont inférieures à borne_basse
# ou supérieures à bornes hautes
# Autrement dit la note tombe en dehors de mon intervalle 

conditionFinale # On constate qu'au moins la ligne avec indice 3 tombe en dehors des bornes

# On finit par extraire de df (grâce à .loc)
# les lignes vérifiant la conditionFinale
# ainsi que toutes les colonnes (grâce aux ":")
df.loc[ conditionFinale , : ]

# Cela apparait déjà dans les boxplot
# en effet les moustaches correspondent par défaut
# aux bornes de cet intervalle
df.note.plot.box()


### --- Statistiques entre deux variables


# On veut identifier les valeurs qui sont négatives parmis cette table numérique

### lien entre deux variables quantitatives

#-- covariance entre deux variables spécifiées
df.calories.cov(df.note)
# Une tendance de décroissante : si calories augmente, la note diminue

# Vérifiez cette tendance graphiquement
df.plot.scatter(x='calories', y='note')

# ou bien entre toutes covariances entre colonnes numériques
df.cov(numeric_only = True)


#-- Cefficient de correlation linéaire
df.calories.corr(df.note)

df.corr(numeric_only = True)

### Exercices :
# La variable avec la plus forte correlation avec note est sucres : -0.762181
df.plot.scatter(x='sucres', y='note')
# On voit que la relation n'est pas complétement linéaire, mais se rapproche d'une droite bruitée
# et lorsque sucres augmente, note diminue (d'où le signe négatif de la correlation)

### -----------------------------------------------------------------


df.groupby('fabricant')
# permet de créer un objet de type GroupBy

df.groupby('fabricant').mean(numeric_only=True)
# Pour chaque modalité du groupe, c'est-à-dire fabricant,
# la moyenne est calculée pour chaque colonne

# La fonction size permet d'obtenir la taille de chaque groupe
df.groupby('fabricant').size()

# même selon plusieurs modalités
df.groupby(['fabricant','type']).size()

# On peut bien évidemment se limiter à certaines colonnes

df.groupby('fabricant')[['calories','sucres','note']].mean()


## Vous pouvez évidemment appliquer plus d'une fonction, à la fois, grâce
## à la méthode agg et en vous limitant à certaines colonnes
df.groupby('fabricant')[['calories','sucres']].agg([np.mean,np.median])

# et donner les noms que vous voulez
df.groupby('fabricant')[['calories','sucres']].agg([('moyenne',np.mean),('médiane',np.median)])




## Fonction apply permet d'appliquer des fonctions sur des groupes

# Permet de trier
df.sort_values(by='note', ascending=False)[:3]

def nPremiers(monDataFrame, n=3, colonne='note'):
    return monDataFrame.sort_values(by=colonne, ascending=False)[:n]

nPremiers(df, n=2, colonne='note')

df.groupby('fabricant').apply(nPremiers, n=2, colonne='note')
# Cela retourne pour chaque fabricant, les céréales avec les meilleurs notes

### Exercices :
df.groupby('étagère')[['calories', 'protéines', 'sucres', 'vitamines', 'potassium','note']].agg([np.mean, np.std])
# On groupe selon les modalités d'étagère, et sur les colonnes calories...potassium, on calcule la moyenne et l'écart-type


### Créer des groupes à partir de valeurs

groupNotes = pd.cut(df.note, bins=[0,25,50,75,100], include_lowest=True)
groupNotes # On obtient un objet de type category

# On peut déterminer le nombre d'éléments dans chaque classe
groupNotes.value_counts()

# On peut aussi définir nos propres labels qui seront utilisés comme noms de classes
groupNotes = pd.cut(df.note, bins=[0,25,50,75,100], include_lowest=True, labels=['mauvais','moyen','bon','excellent'])

groupNotes.value_counts()

df.groupby(groupNotes).mean(numeric_only=True)


### Exercices :

# 1.
# Il est aussi possible de couper selon des quantiles
groupNotesQuartiles = pd.qcut(df.note, 4) # ici selon les 3 quartiles

# 2.
groupNotesQuartiles.value_counts()

# 3.
### Pour revenir à nos groupby
### on peut directement donner le résultat de la fonction cut
df.groupby(groupNotesQuartiles)

df.groupby(groupNotesQuartiles)[['sucres','calories','lipides']].agg([np.mean,np.median,np.std])


### -----------------------------------------------------------------

### Distribution d'une variable

# en effectifs
df['fabricant'].value_counts()

# en fréquence
df['fabricant'].value_counts(normalize=True)

### Exercices :
# 1.
# On s'intéresse à la variable sucres
df.sucres
# Pour savoir les bornes min et max de nos intervalles, regardons
# quelles sont les valeurs min et max de sucres
df.sucres.min()
df.sucres.max()

# On crée des intervalles contenant 0 et 15
groupSucres = pd.cut(df.sucres, bins=[0,5,10,15], include_lowest=True)

# 2.
# On calcule les effectifs dans chaque classe
groupSucres.value_counts()
# Ainsi que les fréquences
groupSucres.value_counts(normalize=True)
    


### Distribution de plusieurs variables

### Distribution jointe en termes d'effectifs

tabEffFabType = pd.crosstab(df.fabricant, df.type, margins=True)
tabEffFabType

### Exercices :
### Distribution jointe en termes de fréquences

tabEffFabType = pd.crosstab(df.fabricant, df.type, margins=True, normalize=True)
tabEffFabType


### Exercices :
### Distribution conditionnelles en termes d'effectives

pd.crosstab(df.fabricant, df.type, margins=True)
# pour la distribution du nombre de produits par fabricant sachant que type=chaudes
# lire sur la colonne chaudes.

# Idem pour fabricant sachant que type = froides

pd.crosstab(df.fabricant, df.type, margins=True, normalize='columns')
# on divide chaque colonne, par la somme des valeurs de la colonne, cela revient
# à calculer la distribution en fréquence de fabricant conditonnelement au type

pd.crosstab(df.fabricant, df.type, margins=True, normalize='index')
# il s'agit ici de la distribution en fréquence, du type sachant le fabricant



###---------- 8. Indépendance entre deux variables qualitatives

# 1. Comparaison des distributions conditionnelles
### il faut que la distribution du fabricant, et les distribution des
### fabricant conditionnellement au type de céréales soient similaire.

distFabCondType = pd.crosstab(df.fabricant, df.type, margins=True, normalize='columns')
distFabCondType
# nous montre que les distributions sont assez différentes.

distFabCondType.plot.bar() # 

# 2. Calcul de la statistique du Chi-2

### Autre méthode pour vérifier l'indépendence entre les deux, est de faire
### un test du Chi-2

# 0. Partons de la table des effectifs

tabEff = pd.crosstab(df.fabricant, df.type)
tabEff

# Chi-square test of independence. 
stat_chi2, p_value, dof, expected = ss.chi2_contingency(tabEff)
# Affichons la statistique du Chi-2
print(stat_chi2)
# Ainsi que la p-value
print(p_value)

# La p-value est très faible (moins de 5%), cela signifie que l'on rejette l'hypothèse null
# d'indépendence.


######################################
### Mesurer la relation entre une variable quantitative et une variable qualitative

# Idée 1 : réprésenter les distributions conditionnelles
# !!! lancer ces trois lignes ensembles
fig, axes = plt.subplots(nrows=1, ncols=2)
df.plot.box(column='note', by='fabricant', rot=90, ax=axes[0])
df.plot.box(column='note', ax=axes[1])


# Idée 2 : calculer la moyenne et variance conditionnelles selon les modalités fabricant
df.groupby('fabricant')['note'].mean()
df.groupby('fabricant')['note'].var(ddof=0)
# A comparer à la version non conditionnelle
df['note'].mean() 
df['note'].var(ddof=0)


# Idée 3 : le coefficient de détermination

### Exercices :

# 1.
# Calcul des Var(y_k), soit les variances de note par groupe, c'est-à-dire par fabricant
varGroup = df.groupby('fabricant')['note'].var(ddof=0)
varGroup

# Calcul des N_k, soit la taille de chaque groupe
countGroup = df.groupby('fabricant').size()
countGroup

# Calcul de la variance intra-groupe
varianceIntra = ( countGroup * varGroup ).sum() / countGroup.sum()

# 2. Calcul de la variance totale
varianceTotale = df['note'].var(ddof=0)

# 3. Calcul du R2
Rsquare = 1 - varianceIntra/varianceTotale
Rsquare
# Interprétations :
# 1 : La note serait entièrement déterminée par la connaissance du fabricant
# 0 : aucune dépendance de la note avec le fabricant
# 0.36835 dépendance très modérées


# Vérification :
# Calculer le R2 = variance inter / variance totale
varNotes = df['note'].var(ddof=0)
meanNotes = df['note'].mean()
meanGroup = df.groupby('fabricant')['note'].mean()

varianceInter = ( countGroup * ( meanGroup - meanNotes )**2 ).sum() / countGroup.sum()
RsquareVerif = varianceInter / varianceTotale
RsquareVerif



##############################
## Régression linéaire

# Nous avons vu que la note et la teneur en sucres présentaient une relation plutôt linéaire
df.plot.scatter(x='sucres', y='note')
# Ajustons une régression linéaire entre ces deux variables

res = ss.linregress(df.sucres,df.note)
res.slope
res.intercept

# Nous avons les a et b estimés
a_hat = res.slope # donne la valeur de a
a_hat
b_hat = res.intercept # donne la valeur de b
b_hat

# Prenons deux points de notre droite étant donné deux valeurs de teneurs en sucres assez éloignées
val_sucres = np.array([0,20])
val_notes = a_hat*val_sucres + b_hat
val_notes

# Traçons la courbe que nous venons de calculer sur notre nuage de points
# y = ax + b

# Traçons notre nuage de points, ainsi que la courbe de régression
# Lancer les deux lignes ci-dessous en même temps
df.plot.scatter(x='sucres', y='note')
plt.plot(val_sucres, val_notes)

### Ce n'est peut être pas la meilleure relation entre y et x !
### Il y a bien d'autres relations que l'on peut imaginer !


### Exercices : Ajuster le modèle de régression linéaire entre la note et le nombre de calories 

df.plot.scatter(x='calories', y='note')

# 1.
res = ss.linregress(df.calories,df.note)
res.slope
res.intercept

# Prenons deux points de notre droite étant donné deux valeurs de teneurs en sucres assez éloignées
a_hat = res.slope # donne la valeur de a
a_hat
b_hat = res.intercept # donne la valeur de b
b_hat

# 2.
nb_calories = np.array([40,180])
val_notes = a_hat*nb_calories + b_hat
val_notes

# 3.
# Traçons notre nuage de points, ainsi que la courbe de régression
# Lancer les deux lignes ci-dessous en même temps
df.plot.scatter(x='calories', y='note')
plt.plot(nb_calories, val_notes)
