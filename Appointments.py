# Dataset https://www.kaggle.com/joniarroba/noshowappointments
import numpy as np
import matplotlib as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import sklearn
import pandas as pd
########################################################################################
data=pd.read_csv("C:\\Users\\Christopher\\Desktop\\Mathematical Medicine\\Appointments.csv")
data=data.iloc[:,2:14]
data=data.drop(['ScheduledDay','AppointmentDay'], axis=1)
X=data.iloc[:,0:9]
Y=data.iloc[:,-1]
#####################################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ENC = LabelEncoder()
ENC.fit(Y)
Y=ENC.transform(Y)
count = 0
for col in X:
    if X[col].dtype == 'object':
        if len(list(X[col].unique())) <= 2:
            ENC.fit(X[col])
            X[col] = ENC.transform(X[col])
            count += 1
X = pd.get_dummies(X)
##################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc=MinMaxScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
###################################################################################
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
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
####################################################################################
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier=Sequential()
classifier.add(Dense(output_dim=100, init='random_uniform', activation='relu', input_dim=89))
classifier.add(Dense(output_dim=200, init='random_uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=100, init='random_uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=1, init='random_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=100, nb_epoch=50)
y_pred=classifier.predict(X_test)
from sklearn.metrics import roc_auc_score
print("ROC AUC score of the neural network classifier is:")
print(roc_auc_score(Y_test, y_pred))
####################################################################################
from sklearn.naive_bayes import GaussianNB, MultinomialNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
print(roc_auc_score(Y_test, y_pred))
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)
####################################################################################
