# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import itertools
import sklearn
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

df=pd.read_csv('diabetes-1.csv')

"""
plt.figure()
plt.scatter(df[df.Outcome==1].BloodPressure,df[df.Outcome==1].Glucose,label='D
iabete',color='r',s=3)
plt.scatter(df[df.Outcome==0].BloodPressure,df[df.Outcome==0].Glucose,label='N
o Diabete',color='b',s=3)
plt.legend()
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
"""

for c in df.columns:
	print('For column',c,'there are',df[df[c]==0][c].count(),'zero values.')

"""
plt.figure()
plt.hist(df[c],bins==15)
plt.xlabel(c)
plt.ylabel('frequency')
plt.show()
"""


df['PredictedOutcome']=np.where(df.Glucose<125,0,1) #np.where(df.Age<30,0,1)
N_correct=df[df.PredictedOutcome==df.Outcome].shape[0]
N_total=df.shape[0]
accuracy=N_correct/N_total
print('Nombre d exemples corrects = ',N_correct)
print('Nombre d exemples total = ',N_total)
print('Precision du Glucose = ',accuracy)


donnees = df.drop('Insulin',axis=1,inplace=False)


train, test = train_test_split(df, test_size = 0.3, random_state = 0)
#train.describe()


def imputeColumns(dataset):
	columnsToImpute=['Glucose', 'BloodPressure', 'SkinThickness','BMI']
	for c in columnsToImpute:
		avgOfCol=dataset[dataset[c]>0][[c]].mean()
		print(avgOfCol)
		dataset[c+'_imputed']=np.where(dataset[[c]]!=0,dataset[[c]],avgOfCol)
		#dataset[c]=np.where(dataset[[c]]!=0,dataset[[c]],avgOfCol)

imputeColumns(train)
imputeColumns(test)

X_train = train[['Pregnancies','Glucose_imputed','BloodPressure_imputed','SkinThickness_imputed','BMI_imputed','DiabetesPedigreeFunction','Age']]
Y_train = train[['Outcome']]
X_test = test[['Pregnancies','Glucose_imputed','BloodPressure_imputed','SkinThickness_imputed','BMI_imputed','DiabetesPedigreeFunction','Age']]
Y_test = test[['Outcome']]


arbre_de_decision = tree.DecisionTreeClassifier(random_state = 0)
arbre_de_decision.fit(X_train, Y_train)
arbre_de_decision.score(X_test, Y_test)


#Les decisions de l'arbre concertant l'ensemble test
resultats_decision_test = arbre_de_decision.predict(X_test)
#Pour simplifier les verifications, nous enregistrons les vrais resultats dans un tableau

tableau_Y_test = Y_test.reset_index(level=None, inplace=False)
#Variables indiquant le nombre de couples selon les 4 cas :
#La première condition etant la decision de l’arbre à V(rai) ou F(aux)
#La seconde etant le reel etat de la patiente
VV = 0
VF = 0
FV = 0
FF = 0
#Boucle verifiant chaque cas des patientes
for i in range(0, resultats_decision_test.shape[0]):
	if (resultats_decision_test[i]==1 and tableau_Y_test.Outcome[i]==1):
		VV = VV + 1
	elif (resultats_decision_test[i]==1 and tableau_Y_test.Outcome[i]==0):
		VF = VF + 1
	elif (resultats_decision_test[i]==0 and tableau_Y_test.Outcome[i]==1):
		FV = FV + 1
	else:
		FF = FF + 1

# Affichage
print("Statistiques (en pourcentage) :")
print("Pourcentage de VV", (VF/resultats_decision_test.shape[0])*100, "%")
print("Pourcentage de VF", (VF/resultats_decision_test.shape[0])*100, "%")
print("Pourcentage de FV", (FV/resultats_decision_test.shape[0])*100, "%")
print("Pourcentage de FF", (FF/resultats_decision_test.shape[0])*100, "%")

#Selection du nombre de voisins le plus proche, ici 5
knn = KNeighborsRegressor()
#Initialisation du knn via sa methode d’apprentissage
knn.fit(X_train, Y_train)
#Effectue les predictions du knn via son modèle
predictions = knn.predict(X_test)

#Calcule le taux d’erreur de la prediction
erreur = (((predictions - Y_test)**2).sum())/len(predictions)
print("Pourcentage de precision de la methode KNN :", (1-erreur.Outcome)*100,"%")


