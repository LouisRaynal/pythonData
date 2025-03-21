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

