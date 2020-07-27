# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:41:22 2020

@author: vijayk1
"""
# LSTM RNN - Time Series Prediction 

import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten,Dense

timeseries_data = [110,125,133,146,158,172,187,196,210]
n_feat = 3

def prepare_data(timeseries_data,n_feat):
    X,y = [],[]
    for i in range(len(timeseries_data)):
        end_ix = i + n_feat
        if end_ix > len(timeseries_data) - 1:
            break
        seq_x,seq_y = timeseries_data[i:end_ix],timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X,y = prepare_data(timeseries_data,n_feat)

# In LSTM RNN - we need to convert numpy array into 3D data.
# The current shape of X is 6,3 which refers to 6 records and 3 timestamps 
# We can call timestamps as features and thus we just add third dimension as features
# 6,3,1

X = X.reshape(X.shape[0],X.shape[1],1)

# Building LSTM RNN model 

model = Sequential()
model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(3,1)))
model.add(LSTM(50,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=200,verbose=0)

X_input = np.array([187,196,210])
X_inp = X_input.reshape(1,3,1)
yhat = model.predict(X_inp,verbose=0)
n_steps = 3
n_features = 1

# demonstrate prediction for next 10 days
x_input = np.array([187, 196, 210])
temp_input=list(x_input)
lst_output=[]
i=0
while(i<10):
    
    if(len(temp_input)>3):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
    

print(lst_output)

import matplotlib.pyplot as plt
day = np.arange(1,10)
day_pred = np.arange(10,20)
plt.plot(day,timeseries_data)
plt.plot(day_pred,lst_output)

