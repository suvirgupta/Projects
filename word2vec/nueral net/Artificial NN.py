import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


churn = pd.read_csv(r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\Artificial_Neural_Networks\Churn_Modelling.csv')

x = churn.iloc[:,3:13].values
y = churn.iloc[:,13].values

## Categorical variable encoding
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

encode_x_1 = LabelEncoder()
x[:,1] = encode_x_1.fit_transform(x[:,1])

encode_x_2 = LabelEncoder()
x[:,2] = encode_x_2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features=  [1] )
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

## splitting the data set into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2 , random_state= 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

## import the keras library to bulit the nueral network

import keras
from keras.models import Sequential
from keras.layers import Dense

neuronbuild = Sequential()
Dense
neuronbuild.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu', input_dim = 11) )
neuronbuild.add(Dense(units = 6, kernel_initializer = 'uniform', activation='relu') )
neuronbuild.add(Dense(units = 1, kernel_initializer = 'uniform', activation='sigmoid') )

neuronbuild.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
neuronbuild.fit(x_train,y_train,batch_size= 10 , nb_epoch= 100)

y_pred = neuronbuild.predict(x_test)
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)