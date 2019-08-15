# Dataset https://www.kaggle.com/uciml/pima-indians-diabetes-database
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
#################################################################################
data=pd.read_csv("C:\\Users\\Christopher\\Desktop\\diabetes.csv")
X=data.iloc[:,0:8]
Y=data.iloc[:,-1]
#####################################################################################
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc=MinMaxScaler()
X=sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
####################################### Artificial Neural Network #################################################
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=40, init='random_uniform', activation='relu', input_dim=8))
classifier.add(Dense(output_dim=40, init='random_uniform', activation='relu'))
classifier.add(Dense(output_dim=40, init='random_uniform', activation='relu'))
classifier.add(Dense(output_dim=40, init='random_uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='random_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=10, nb_epoch=150)

y_pred=classifier.predict(X_test)
from sklearn.metrics import roc_auc_score
print("ROC AUC score of the neural network classifier is:")
print(roc_auc_score(Y_test, y_pred))
####################################### Gradient-Boosting Machine ################################################
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
seed = 7
model = XGBClassifier()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy of the Gradient Boosting Machine is: %.2f%%" % (accuracy * 100.0))
print("ROC score of the gradient Bossting Machine is:" )
print(roc_auc_score(Y_test, y_pred))
################################################################################################################
