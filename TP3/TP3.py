# -*- coding: utf-8 -*-
"""
TP 3 d'analyse de données avec Python
"""

# RAYNAL Louis

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

pd.options.display.max_columns = None


#-------------- Chargement et nettoyage du jeu de données --------------

# Question 2
# Changer de répertoire
os.chdir('/Users/Guest/Desktop')
#os.chdir('D:/Louis/Documents/Cours_AnalyseDeDonneesPython/TP3')
# Vérifier où l'on se trouve
os.getcwd()
# Lister les fichiers du répertoire
os.listdir()
# On stocke le nom du fichier dans une variable, nomFichier
nomFichier = 'titanic.csv'
# On lit le fichier dont le contenu est stocké dans un DataFrame nommé df
df = pd.read_csv(nomFichier, header=0, skipfooter=1, sep=';')

# Question 3
# Vérification
df
# ou
df.shape

#------------- Nettoyage du jeu de données --------------


# Question 4
# Afficher les 10 premières lignes
df.head(10)
# Le dixième passager est de deuxième classe et a survécu

# Question 5
# Pour avoir les noms de colonnes actuels
df.columns

# Création du dictionnaire avec les anciens et nouveaux noms à utiliser
dicoNomCols = {'PassengerId':'idPassager', 'Survived':'survie',
               'Pclass':'classe', 'Name':'nom',
               'Sex':'sexe', 'Age':'age',
               'SibSp':'nbFreresEpoux', 'Parch':'nbParentsEnfants',
               'Ticket':'noTicket', 'Fare':'prixTicket',
               'Cabin':'noCabine', 'Embarked':'portEmbarquement'}
# Sélectionnez bien toute la définition du dictionnaire avant de le lancer

# On renomme les colonnes
df.rename(columns=dicoNomCols, inplace=True)

#Alternative :
#df.columns = ['idPassager','survie','classe','nom','sexe','age','nbFreresEpoux','nbParentsEnfants','noTicket','prixTicket',
#              'noCabine','portEmbarquement']


# Vérification
df.columns


# Question 6
# On compte le nombre de lignes dédoublonnées
df.duplicated().sum()
# il y a 2 lignes doublons

# On supprime les lignes dédoublonnées
df.drop_duplicates(inplace=True)

# On vérifie qu'il n'y a bien plus de doublons
df.duplicated().sum()


# Question 7. (a)
# Vérifier pour chaque colonne combien de valeurs manquantes il y a (s'il y en a)
df.isnull().sum(axis=0)

# Question 7. (b)
### - On veut supprimer la colonne avec le numéro de cabine étant donné tous les doublons présents
df.drop(columns = 'noCabine', inplace=True)

# Question 7. (c)
# Attention, pour récupérer la valeur du mode, il vous faudra ajouter [0] à la fin de votre ligne de code
# Le résultat devrait être 'S'
df.portEmbarquement.mode()[0] # le mode est S

dicoRemplacementPort = {'portEmbarquement':df.portEmbarquement.mode()[0]}
df.fillna(dicoRemplacementPort, inplace=True)

# Vérification
df.loc[ ( df.idPassager == 62 ) | ( df.idPassager == 830 ) , : ]
# Alternative plus simple
df.loc[ df.idPassager.isin([62,830]) , : ]
# Autre alternative : 1 par 1
df.loc[ df.idPassager == 62 , : ]
df.loc[ df.idPassager == 830 , : ]

df


# Question 7. (d)
### Pour la colonne avec l'âge, étant donné le nombre important de vide
### supprimons les vides
df.dropna(inplace=True)
# Alternative : spécifier grâce au nom de la colonne où se trouve les NaN
df.dropna(subset=['age'], inplace=True)
# Ici ces deux méthodes donnent le même résultats
# car les seuls NaN qui restent se trouvent dans la colonne 'age'

# Question 7. (e)
### Vérifiez que vous n'avez plus de valeurs manquantes
df.isnull().sum(axis=0) # ok !
df

# Question 8
# C = Cherbourg, Q = Queenstown, S = Southampton
dicoPort = {'C':'Cherbourg','Q':'Queenstown','S':'Southampton'}
df.portEmbarquement = df.portEmbarquement.map(dicoPort)

df.portEmbarquement.unique() # ok !


#------------- Etude de l'âge des passagers --------------

# Question 9
# Age moyen et médian
df.age.mean()
df.age.median()
# Ecart-type    
df.age.std(ddof=0)

# Question 10
# Age minimum et maximum
df.age.min()
df.age.max()
# Etendu
np.ptp(df.age)

# Question 11
df.age.plot.hist(bins=20, density=True)
# Les âges semblent distribués de manière normale, centrée autour de 30 ans (la moyenne)
# avec néanmoins un pic d'âges proches de 0

# Question 12
df.plot.box(column='age', rot=70)
# A la lecture du 1er et 3ieme quartile, nous pouvons dire que 50% des passagers ont un âge
# entre 20 et 40 ans.
# Bonus :
df.age.quantile(q=0.25)
df.age.quantile(q=0.75)

# Question 13
# - Quel est l'âge du plus vieux passager ? Et a-t-il survécu ? (utiliser la méthode max)
df.age.max() # voici l'âge maximal des passagers
# On récupère les lignes égales aux max , et toutes les colonnes
df.loc[ df.age == df.age.max() , : ]
# Il a survécu !

# Bonus : pour le plus jeune
df.age.min()
df.loc[ df.age == df.age.min() , : ]


#------------- Étude de la survie des passagers --------------


# Question 14
# Il suffit de faire la moyenne de la colonne survie.
df.survie.mean() # 40.6% de survie
# Autre manière, avec la distribution de la colonne survie, en fréquence
df.survie.value_counts(normalize=True)

### Nous voulons maintenant savoir si les femmes ont une plus grande chance de
### survie que les hommes

# Calculer le taux de survie selon les modalités de la variable sexe

# Question 15
df.groupby('sexe')['survie'].mean() # la survie des femmes est de 75%, contre 20% pour les hommes

# Question 16
# Regardons aussi l'effet selon la classe
df.groupby('classe')['survie'].mean() 
# la classe 1 à 66% de survie
# contre 48% pour la classe 2 et 24% pour la classe 3

# Question 17
# Regardons maintenant selon des classes d'âges,
# nous voulons savoir si les personnes de moins de 20 ans ont plus de chance de survie
groupJeunesVieux = pd.cut(df.age, bins=[0,20,80], labels=['jeunes','vieux'], include_lowest=True)
df.groupby(groupJeunesVieux)['survie'].mean() # 46% de survie parmi les 20 ans ou moins

# Question 18
# Tester l'indépendance entre survie et sexe
tabSurvieSexe = pd.crosstab(df.survie, df.sexe)
tabSurvieSexe

stat_chi2, p_value, dof, expected = ss.chi2_contingency(tabSurvieSexe)
print(stat_chi2) # La statistique du Chi-2 est trés éloignée de 0
print(p_value) # On rejette l'hypothèse d'indépendance entre les deux variables car la p-value est inférieure à 5%
# Les variables sexe et survie ne sont pas indépendantes

# Question 19
# Tester l'indépendance entre survie et les classes d'âges [0-20], ]20-80]
tabSurvieAge = pd.crosstab(df.survie, groupJeunesVieux)
tabSurvieAge

stat_chi2, p_value, dof, expected = ss.chi2_contingency(tabSurvieAge)
print(stat_chi2) # La statistique du chi-2 n'est pas très élevée
print(p_value)
# La p-value n'est pas inférieure à 5%, on a donc indépendance entre la survie
# et ces deux classes d'âges
# Remarque : si on avait changé les classes d'âges pour [0,16],]16,80]
# là nous aurions eu une dépendance.

# Question 20
### Pour finir calculer le taux de survie selon les modalités conjointe de la variable sexe et classe
### Quel est le taux de survie des femmes de première classe ?
df.groupby(['sexe','classe'])['survie'].mean()
# Les femmes de première classe ont un taux de survie de 96% !

#------------- Étude du prix du ticket --------------


# Question 21
### Histogramme des prix des tickets, en 10 classes, et avec la densité de fréquence en ordonnée
df.prixTicket.plot.hist(bins=10, density=True)
# On remarque que la majorité des prix de ticket sont dans la classe [0-50]
# mais qu'il existe tout de même certains passagers qui ont payés un prix très élevé allant jusqu'à 500 livres

# Question 22
df.prixTicket.mean()
df.prixTicket.median()
# Cette forte différence est provoquée par les prix très élevés payés par certains passagers.
# En effet, la moyenne est tirée vers le haut à cause de ces hautes valeurs
# alors que la médiane est insensibles à ces valeurs extrèmes

# Bonus :
df.prixTicket.std(ddof=0)

df.prixTicket.min()
df.prixTicket.max()
np.ptp(df.prixTicket)
df.plot.box(column='prixTicket')

# Question 23
df.plot.scatter(x='age', y='prixTicket')
# A la vue de ce graphique, est-ce que lorsque l'âge augmente, le prix du ticket semble augmenter ?
# Pas vraiment. Le calcul de la covariance pourrait nous éclairer.

# Question 24
# Calculer la covariance entre l'âge et le prix du ticket
# Interpréter ce résultat.
df.age.cov(df.prixTicket)
# D'après cette covariance, lorsque l'âge augmente, le prix du ticket aurait tendance à augmenter

# -------------------
## Bonus (hors questions)
# Pour voir plus clairement le lien entre l'âge et le prix du ticket,
# nous allons créer des groupes d'âges [0,20], ]20,40], ]40,60] et ]60,80]
groupAges = pd.cut(df.age, bins=[0,20,40,60,80])

# Sur chaque classe d'âge, calculer l'effectif puis la fréquence dans un deuxième temps
groupAges.value_counts().sort_index()
groupAges.value_counts(normalize=True).sort_index()
# Note : vous pouvez trier les résultats par classe avec la méthode sort_index()

# Calculer maintenant la moyenne et la médiane des prix du Ticket par groupe
df.groupby(groupAges)['prixTicket'].agg(func=[np.mean,np.median])
# -------------------


# Question 25
### prix = a * age + b + epsilon
res = ss.linregress(df.age, df.prixTicket)
a_hat = res.slope
a_hat
b_hat = res.intercept
b_hat

# Question 26
val_ages = np.array([0,100])
val_prix = a_hat * val_ages + b_hat
val_prix
# Pour 0 an : 24 livres
# Pour 100 ans : 59 livres

# Question 27
### Utiliser ces estimations pour visualiser la droite de regression sur le nuage de points entre age et prix
df.plot.scatter(x='age', y='prixTicket')
plt.plot(val_ages, val_prix, c='r') # c='r' pour couleur rouge

# Question 28
### Pensez-vous que le modèle linéaire décrit bien la relation
### entre l'âge et le prix du ticket ?
### Calculer la correlation entre ces deux variables pour appuyer vos propos.
df.age.corr(df.prixTicket)
# La correlation est de 0.1, ce qui décrit une relation faiblement linéaire entre nos deux variables.
# Le modèle linéaire n'est pas adapté ici.

# Question 29
### Nous voulons maintenant évaluer la relation entre la variable classe 
### et le prix du ticket

# Question 29. (a)
fig, axes = plt.subplots(nrows=1, ncols=2)
df.plot.box(column='prixTicket', by='classe', ax=axes[0])
df.plot.box(column='prixTicket', ax=axes[1])
# Il semble y avoir une différence du prix du ticket selon la classe, en particulier
# entre la première et les deux autres classes

# Question 29. (b)
### Avec le calcul du R2
varGroup = df.groupby('classe')['prixTicket'].var(ddof=0)
varGroup

countGroup = df.groupby('classe').size()
countGroup

varianceIntra = ( countGroup * varGroup ).sum() / countGroup.sum()

varianceTotale = df['prixTicket'].var(ddof=0)

Rsquare = 1 - varianceIntra/varianceTotale
Rsquare
# 0.36 pour le R2, on peut dire que le prix du ticket est modérément impacté
# par la classe des passagers.
# Peut être que si nous avions créé deux groupes :
# "Première classe" vs "Hors première classe" nous aurions eu un R2 encore plus élevé
# car la différence de prix semble surtout provenir de la première classe.
