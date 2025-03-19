# -*- coding: utf-8 -*-
"""
TP 1 d'analyse de données avec Python
"""

### Import des librairies

import numpy as np
import pandas as pd

###-------- Rappels :


## Les tuples   

tuple_1 = (1, 'a', True)
tuple_1

tuple_1[2]

#tuple_1[2] = 'b'

tuple_2 = tuple([1,2,3])
tuple_2

tuple_2[1]

## Les listes

liste_1 = ['abc', 'def', 'ghi']
liste_1[1]

liste_1[2] = 0

liste_2 = list(tuple_1)
liste_2

## Les dictionnaires

dico_1 = dict(a=1, b=2, info='des informations')
dico_1['a']

dico_2 = {'a':1, 'b':[1,2,3,4,5], 'c':"plus d'informations"}
dico_2['b']

dico_2['b'] = 5
dico_2['b']

dico_2.keys()
dico_2.values()

###---------- Numpy

# création de vecteurs/matrices
data_1 = [1.5,2,3.4,4,5,6,7,8]  # à partir d'une liste

array_1 = np.array(data_1)
array_1

data_2 = [[1,2,3,4], [5,6,7,8]] # à partir d'une liste de listes
array_2 = np.array(data_2)
array_2

# Obtenir les dimensions de ces deux arrays
array_1.shape
array_2.shape

# Obtenir le type de ces deux arrays
array_1.dtype
array_2.dtype

# Autres fonctions pour créer des arrays
np.zeros(5)
np.zeros((5,3))

np.ones(5)
np.ones((5,3))

np.eye(5)

np.arange(20)

# Il est possible lors de la création d'un array
# de spécifier un type

array_f = np.array([1,2,3], dtype=np.float64)
array_i = np.array([1,2,3], dtype=np.int32)

array_f.dtype
array_i.dtype

array_f
array_i

# Calculs
array_2 + array_2
array_2 * array_2
array_2 + 42

# Comparaisons
array_1 > 5
array_1 == 2
array_1 != array_1

# Récupérer des parties d'arrays (1D)
array1D = np.arange(5,10)
array1D

# Différentes manières de faire
array1D[:]
array1D[1]
array1D[1:3]
array1D[1:]
array1D[:4]
array1D[-1]
array1D[-3:]
array1D[[0,4,2]]

# Modifications
array1D[0:2]=17
array1D

# Idem mais pour un array en 2D
array2D = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
array2D

# Extraction de lignes
array2D[1,:]        # équivalent à array2D[1]
array2D[[1,2],:]    # équivalent à array2D[[1,2]]
array2D[:2,:]       # équivalent à array2D[:2]
# Le : pour dire "toutes les colonnes" n'est pas obligatoire

# Extraction de colonnes
array2D[:,1]
array2D[:,[0,1]]
array2D[:,1:]
# Le : pour dire "toutes les lignes" est obligatoire

# Extraction de lignes et colonnes
array2D[1][2] # soit en deux temps
array2D[1, 2] # soit d'un coup

# Attention !
array2D[[0,1],[0,3]] 
# donne les éléments avec indices (0,0) et (1,3)
# et non, tous les éléments aux lignes d'indice 0 et 1,
# et colonnes d'indice 0 et 3
# Pour ce faire, vous pouvez utiliser la fonction np.ix_
# de la manière suivante :
array2D[np.ix_([0,1],[0,3])]


# Modifications
array2D[[0,1],[0,3]] = 0
array2D

# Récupération de sous-ensembles via des conditions
array2D > 5                 # Identifie par True les éléments > 5, False sinon
array2D[ array2D > 5 ] = -1 # On remplace ces valeurs > 5 par -1
array2D


#--------------------------------------------
# Exercices :
# 1.
array_ex = np.zeros((6,4))#, dtype=np.int32)
array_ex

# 2.
array_ex[[2,4],:]=[1,2,3,4]
array_ex

# 3.
array_ex[array_ex == 0] = 10
array_ex

# 4.
array_ex[np.ix_([1,2],[0,2,3])] = 1.5
array_ex

# 5. Relancer en ajoutant dtype=np.int32
array_ex = np.zeros((6,4), dtype=np.int32)
array_ex
array_ex[[2,4],:]=[1,2,3,4]
array_ex
array_ex[array_ex == 0] = 10
array_ex
array_ex[np.ix_([1,2],[0,2,3])] = 1.5
array_ex
# On remarque que cet array n'accepte que des valeurs entières (int)
# les nombres décimaux sont tronqués pour n'avoir que la partie entière
#--------------------------------------------


### !!! Attention !!!
array_1 = np.array([1,2,3,4])
array_copie = array_1[0:2]
array_copie

array_copie[1] = 42
array_copie
array_1

# Lors d'une copie d'un array
array_1 = np.array([1,2,3,4])
array_copie = array_1[0:2].copy()
array_copie

array_copie[1] = 42
array_copie
array_1    

# -------- Présentation de Pandas ---------

# Créons un DataFrame avec trois colonnes à partir d'un dictionnaire
produit = ['Corn Flakes','Crispix','Golden Grahams','Muesli Raisins','Smacks','Special K']
calories = [100,110,110,150,110,110]
proteines = [2,2,1,4,2,6]

data = {'Produit' : produit,
        'Calories' : calories,
        'Proteines' : proteines}
# Exécutez bien les trois lignes ci-dessus en une fois

df = pd.DataFrame(data)
df

# Idem mais à partir d'une liste de liste
data2 = [['Corn Flakes',100,2],
         ['Crispix',110,2],
         ['Golden Grahams',110,1],
         ['Muesli Raisins',150,4],
         ['Smacks',110,2],
         ['Special K',110,6]]

df2 = pd.DataFrame(data2, columns=['Produit','Calories','Proteines'])
df2

# Il est enfin possible d'accéder uniquement aux valeurs du DataFrame
df.values # remarquez de c'est un objet de type numpy array qui est retourné

df.columns

df.index

df.dtypes

# Renommer colonnes et lignes
df.columns = ['Produit','Calories','Protéines']

df.index = ['un','deux','trois','quatre','cinq','six']

df # vérification

### -----------------


# Afficher un certain nombre de première lignes / colonnes
df.head()

df.tail()


# Extraction de colonnes
df.Calories
df['Calories']
df[['Produit','Calories']]

# Extraction de lignes
df[2:5]
df[:5]
df[3:]

# Extraction de lignes et colonnes
df.Produit[1:]
df['Calories'][:4]
df[['Produit','Calories']][2:5]

# Il est aussi possible d'utiliser .iloc
# pour sélectionner des lignes et colonnes du DataFrame
# en se basant sur les indices de lignes et colonnes
df.iloc[2:5, :]     # équivalent à df.iloc[2:5]
df.iloc[:, [0,2]]
df.iloc[2:5, [0,2]]
df.iloc[2:5, 2:]
df.iloc[[0,4], [0,2]]

# ne pas confondre df.loc avec df.iloc
# df.loc permet de sélectionner des lignes, colonnes
# en se basant sur les noms de colonnes et de lignes
df.loc[['deux','trois','un'], :]   # équivalent à df.loc[['deux','trois','un']]
df.loc[:, ['Produit','Protéines']]
df.loc[['deux','trois','un'], ['Produit','Protéines']]

# Vous pouvez vous baser sur ce genre d'extractions
# pour changer des valeurs du DataFrame
df.loc[:,'Calories'] = 110
df

df.Protéines == 2   # permet d'identifier dans la colonne Protéines, les éléments égaux à 2
df.loc[df.Protéines == 2,'Protéines'] = 1
df

df.iloc[2:,1] = 120
df

### Fonctions utiles

# Trier par nom de lignes ou colonnes
df.sort_index(axis=0)

df.sort_index(axis=1)

df.sort_index(axis=0, ascending=False)

# Trier selon les valeurs
df.sort_values(by='Protéines')

df.sort_values(by=['Protéines','Produit'])

# Rang dans le classement des valeurs
df.rank(method='min')

help(df.rank)

# df.describe() # peut être utilisé sur une chaine de caractère aussi
# df.count()

df.Protéines.unique()

df.Protéines.count()

df.Protéines.value_counts()


#--------------------------------------------
# Exercices :
# 1.
df2.sort_values(by=['Produit'], ascending=False)

# 2.
df2.sort_values(by=['Calories'], ascending=False).head(1)

# 3.
df2.Calories.value_counts()

# 4.
df2.Calories.value_counts()/df2.Calories.count()
# ou bien
df2.Calories.value_counts(normalize=True)
#--------------------------------------------








