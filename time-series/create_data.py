import numpy as np
def create_data(data, look_back=1):
  dataX, dataY = [],[]
  for i in range(len(data)-look_back-1):
    x = data[i:(i+look_back),0]
    dataX.append(x)
    dataY.append(data[i+look_back,0])
  return np.array(dataX), np.array(dataY)