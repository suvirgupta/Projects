import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


churn = pd.read_csv(r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\Churn_Modelling.csv')

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


### to remove bias and variance trade off we use K-fold cross validation using keras scikit leran wrapper class

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_classifier() :
    neuron = Sequential()
    neuron.add(Dense(unit = 6 , input_dim = 11, kernel_initializer= 'uniform', activation='relu'))
    neuron.add(Dropout(p= 0.2))
    neuron.add(Dense(unit=6, kernel_initializer='uniform', activation='relu'))
    neuron.add(Dense(unit=1, kernel_initializer='uniform', activation='sigmoid'))
    neuron.compile(optimizer = 'adam', loss = 'binary_crossentorpy', metrics = ['accuracy'] )
    return neuron

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10 , nb_epoch = 100)
accuracies = cross_val_score(estimator= classifier, X= x_train, y= y_train, cv = 10 , n_jobs = -1 )

mean = accuracies.mean()
stdev = accuracies.std()


### finding the accur9ate value of the batch size and the epoch for maximizing the prediction accuracy
## using GridSearchCV grid search crossvalidaton optimzation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def keras_classify(optimizer):
    neuron = Sequential()
    neuron.add(Dense(units = 6 , input_dim = 11, activation = 'relu', kernel_initializer = 'uniform'))
    neuron.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    neuron.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    neuron.compile(optimizer = optimizer , loss = 'binary_crossentropy' ,metrics = ['accuracy'])
    return neuron
kerasClassify = KerasClassifier(build_fn= keras_classify)
parameters = {'optimizer':['adam', 'rmsprop'],
              'nb_epoch' :[100,500],
              'batch_size' : [25,32],
              }

grid_search = GridSearchCV(estimator= kerasClassify, param_grid=parameters,scoring = 'accuracy', cv = 10 )
grid_search = grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


best_parameters


best_accuracy









