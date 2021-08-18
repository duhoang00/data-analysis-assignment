#%% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from create_data import create_data

#%% Load data
data = pd.read_csv('./data/airline-passengers.csv', usecols=[1])
plt.plot(data)
plt.show()

#%% Normalize data
sc = MinMaxScaler(feature_range=(0,1))
data = sc.fit_transform(data)

#%% Split data into train, test
train_ = int(len(data)*0.7)
test_ = len(data) - train_
train = data[0:train_,:]
test = data[train_:len(data),:]
print(len(train), len(test))

#%% Reshape data into X = t, Y = t + 1
look_back = 1
trainX, trainY = create_data(train, look_back)
testX, testY = create_data(test, look_back)

#%% reshape input to [samples, time_steps, features]
trainX = np.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
print(testX)
#%% create model
model = Sequential()
model.add(LSTM(4, input_shape=(1,look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX,trainY,epochs=25,batch_size=1)
#%% Predict Test
y_pred = model.predict(testX)
sc.scale_
#%% Inverse prediction
y_pred = sc.inverse_transform(y_pred)
#%% Reshape Baseline
testY = testY.reshape(-1,1)
testY = sc.inverse_transform(testY)
#%% plot baseline and prediction
plt.plot(testY)
plt.plot(y_pred)
plt.show()