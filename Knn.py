import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import torch as th
 


#1) Chargement des données
 
data = pd.read_csv("bank_train_data.csv", sep=",")
label = pd.read_csv("bank_train_labels.csv")
 
 #2)
#X1 = pd.get_dummies(X)
data2 = pd.get_dummies(data)


#Fonction euclidian_distance(v1,v2).
def euclidian_distance(v1,v2):

    distance = 0
    for i in range(v1.shape[0]):
        distance += (v1[i] - v2[i])**2

    return np.sqrt(distance)

print(euclidian_distance(data2.iloc[0],data2.iloc[10]))


#Prenons  un échantillon aléatoire de 60% qui servira d’entraînement
X_train=data2.sample(frac=0.6,random_state=90) 
X_test=data2.drop(X_train.index)

y_train=label.sample(frac=0.6,random_state=90) 
y_test=label.drop(y_train.index)
#print(y_train0.columns)

label_train = y_train['has suscribed']
label_test  =  y_test['has suscribed']

#prendre quelque ligne
label_train_2 = label_train[0:800]
label_test_2  = label_test[0:800]

X_train_2 = X_train[0:800]
X_test_2  = X_test[0:800]




#Fonction neighbors qui renvoie les k plus proches voisins du test.
def neighbors(X_train, y_label, x_test, k):
    list_distances =  []

    for i in range(X_train.shape[0]):

        distance = euclidian_distance(X_train.iloc[i], x_test)

        list_distances.append(distance)


    df = pd.DataFrame()

    df["label"] = y_label
    df["distance"] = list_distances

    df = df.sort_values(by="distance")

    return df.iloc[:k,:]

k = 2
#nearest_neighbors = neighbors(X_train1, label_train1, X_train1.iloc[10], k)
#print(nearest_neighbors)



#Fonction prediction renvoie si l’indivu vas suscrire ou non
def prediction(neighbors):
    
    mean = neighbors["label"].mean()

    if (mean < 0.5):
        return 0
    else:
        return 1

k = 3
nearest_neighbors = neighbors(X_train_2, label_train_2, X_train_2.iloc[10], k)
    
p1 = prediction(nearest_neighbors)

print("La prediction : ", p1)




#Afficher le taux de réussite du classement.
reussite=0
for i in range(X_train_2.shape[0]):
    nearest_neighbors = neighbors(X_train_2, label_train_2, X_train_2.iloc[10], k)
    if (prediction(nearest_neighbors) == label_test_2.iloc[i]): 
        reussite+=1
 

 

#Prenons  un échantillon aléatoire de 60% qui servira d’entraînement
bTest  = pd.read_csv("bank_test_data.csv", sep=",")
bTest2  = pd.get_dummies(bTest)

data_train=bTest2.sample(frac=0.6,random_state=90) 
data_test=bTest2.drop(bTest2.index)


data_train_2 = X_train[0:800]
data_test_2  = X_test[0:800]

k=18
for i in range(bTest2.shape[0]):
    nearest_neighbors1 = neighbors(data_train_2, label_train_2, bTest2.iloc[i], k)
    y_pred = prediction(nearest_neighbors1)
print(y_pred)
np.savetxt('test_results3.csv',y_pred)