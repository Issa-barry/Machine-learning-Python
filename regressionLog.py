import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
#import bigfloat


#1)

data = pd.read_csv("bank_train_data.csv", sep=",")
label = pd.read_csv("bank_train_labels.csv")

#2)
data2 = pd.get_dummies(data)

N=data2.shape[0]
d = data2.shape[1]
weights = np.random.randn(d+1)
print(weights)
#Ajout d'une première colonne de "1" à la matrice X des entrées
data2 = np.hstack((data2,np.ones((N,1))))

#Sortie du modèle
def sigmoid(z):
    return 1 / (1+np.exp(-z))
 

def output(X_train,weights):
	val = sigmoid(np.dot(X_train,weights))
	return val



f = output(data2,weights)
 

#la prediction
def prediction(f):
	return f.round()

y_pred = prediction(f)
#print(y_pred)

#Le taux d'erreur
def error_rate(y_pred,y):
	return (y_pred.reshape(y_pred.shape[0],1) != y).mean()

error = error_rate(y_pred,label)
print(error)
#Le cross entropy
def binary_cross_entropy(f,y):
    return - (y*np.log(f)+ (1-y)*np.log(1-f)).mean()

#le gradiant
def gradient_dot(f,y,X):
	grad = -np.dot(np.transpose(X),(y-f))/X.shape[0]
	return grad

#seperation des donnees en donnees de test 30% et d'entrainement 70%
indices = np.random.permutation(data2.shape[0])

train_idx,test_idx = indices[:int(data2.shape[0]*0.7)], indices[int(data2.shape[0]*0.7):]

X_train = data2[train_idx,:]
y_train = label.iloc[train_idx,:]
X_test = data2[test_idx,:]
y_test = label.iloc[test_idx,:]

#taux d'apprentissage
eta = 0.01

# apprentissage du modele et calule de la performance a chaque 100 iterations
nbepochs = 1000
for i in range(nbepochs):
	f_train = output(X_train,weights)
	y_pred_train = prediction(f_train)

	grad = gradient_dot(f_train,y_pred_train,X_train)

	weights = weights - eta*grad

	if(i%100 == 0):
		error_train = error_rate(y_pred_train,y_train)
		loss = binary_cross_entropy(f_train,y_pred_train)

		f_test = output(X_test,weights)
		y_pred_test = prediction(f_test)
		error_test = error_rate(y_pred_test,y_test)

		print("iteration: " + str(i) + " error train: " + str(error_train) + " loss: " + str(loss) + " error test: " + str(error_test))


# Test avec bank_test_data et sauvegarde du resultat dans bank_test_results
data_train = pd.read_csv("bank_test_data.csv", sep=",")

data_train = pd.get_dummies(data_train)

data_train = np.hstack((data_train,np.ones((data_train.shape[0],1))))

data_test = output(data_train,weights)

y_pred = prediction(data_test)

np.savetxt('bank_test_results.csv',y_pred)






#20) Lancement du modèle avec la librairie scikit-learn et affichage des résultats
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred_test = model.predict(X_test)

error_test = error_rate(y_pred_test, y_test)

print("error_test")
print(error_test)

print("Valeurs des poids du model avec scikitlearn")
print(model.intercept_, model.coef_)
