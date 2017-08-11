import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM


### Extract only the open stock data as array from the google stock dataframe
series_train = pd.read_csv(r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\Recurrent_Neural_Networks\Google_Stock_Price_Train.csv')
series_train = series_train.set_index(series_train.Date)
series_train = series_train.iloc[:,1:2].values
len(series_train) ## 1258

### preprocessing
from sklearn.preprocessing import MinMaxScaler
## Earlier we standarized data in ANN using StandardScaler class now we Normalize it using MinMaxClass
scale = MinMaxScaler()  ### Normalize data (value - Min(array))/ (Max(array)- Min(array)
series_train = scale.fit_transform(series_train)


## Define train and the test set
x_train = series_train[:1257]
y_train = series_train[1:1258]

#### input shape and outshape of the rnn tensor is given below
# Input shapes
# 3D tensor with shape (batch_size(in our example it is the no of observations '1257'), timesteps(is '1' one step in the time ),
# input_dim( '1' one predictor variable open stock price)),
# (Optional) 2D tensors with shape  (batch_size, output_dim).
#
# Output shape
# if return_state: a list of tensors. The first tensor is the output. The remaining tensors are the last states, each with shape  (batch_size, units).
# if return_sequences: 3D tensor with shape  (batch_size, timesteps, units).
# else, 2D tensor with shape (batch_size, units).


x_train= np.reshape(x_train,(1257,1,1))
## defining the RNN neural network

lstm_neuron = Sequential()

lstm_neuron.add(LSTM(units = 4, activation = 'sigmoid',  input_shape =(None,1) ))
lstm_neuron.add(Dense(units=1))
lstm_neuron.compile(optimizer= 'adam', loss = 'mean_squared_error')

lstm_neuron.fit(x_train,y_train,batch_size = 32, nb_epoch=100, )

#### setting up prediction on the test set
## load the test set

series_test = pd.read_csv(r'C:\Users\Suvir Gupta\PycharmProjects\Projects\word2vec\nueral net\data\Google_Stock_Price_Test.csv')
series = series_test.iloc[:,1:2].values

x_test = scale.fit_transform(series)
x_test = np.reshape(x_test, (20,1,1))

y_pred = lstm_neuron.predict(x_test)
### below is the converted predicted values
y_pred = scale.inverse_transform(y_pred)
y_pred.shape


import matplotlib.pyplot as plt

# Visualising the results
plt.plot(series, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Part 4 - Evaluating the RNN
### prediction rmse thought seems very efficient the ouput cannot be considered good
## as the lstm predicts only one day in the advance, hence it won't be give very good prediction for the higher time steps
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(series, y_pred))



## input preprocessing to train on data for last 20 days to get much better predictions
X_train = []
Y_train = []
for i in range(20, 1258):
    X_train.append(series_train[i-20:i, 0])
    Y_train.append(series_train[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train.shape
X_train = np.reshape(X_train,(1238,20,1))


