# Dataset https://data.world/health/breast-cancer-wisconsin
import numpy as np
import pandas as pd
import sklearn
data=pd.read_csv('C:\\Users\\Christopher\\Desktop\\Deep Learning\\health-breast-cancer-wisconsin\\breast-cancer-wisconsin-data\\data.csv')
from sklearn.preprocessing import LabelEncoder
X=data.iloc[:,2:32]
Y=data.iloc[:,1]
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
##################### SUPPORT-VECTOR-MACHINE ###########################
from sklearn.svm import SVC
SVM_CLF=SVC()
SVM_CLF.fit(X_train,Y_train)
ACC=SVM_CLF.score(X_test,Y_test)
print("Accuracy of the support vector machine classifier is:")
print(ACC)
ypred=SVM_CLF.predict(X_test)
from sklearn.metrics import roc_auc_score
print("ROC AUC score of the support vector machine classifier is:")
print(roc_auc_score(Y_test, ypred))
##################### ARTIFICIAL NEURAL NETWORK ###########################
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=6, init='random_uniform', activation='relu', input_dim=30))
classifier.add(Dense(output_dim=6, init='random_uniform', activation='relu'))
classifier.add(Dense(output_dim=6, init='random_uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='random_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=25, nb_epoch=20)
y_pred=classifier.predict(X_test)
from sklearn.metrics import roc_auc_score
print("ROC AUC score of the neural network classifier is:")
print(roc_auc_score(Y_test, y_pred))
