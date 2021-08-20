#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima

#%% Import dataset
data = pd.read_csv('./data/Sunspots.csv')
print(data)

#%% Data overview
plt.plot(data['Monthly Mean Total Sunspot Number'])

#%% Check stationary
stationary_test = adfuller(data['Monthly Mean Total Sunspot Number'])
print(stationary_test)

#%% PACF
pacf = plot_pacf(data['Monthly Mean Total Sunspot Number'])

#%% AR model
AR = ARIMA(data['Monthly Mean Total Sunspot Number'], order = (1,0,0)).fit()
print(AR.summary())

#%% AR visuallize
pred = AR.predict()
plt.plot(data['Monthly Mean Total Sunspot Number'])
plt.plot(pred, color ='red')

#%% ACF
acf = plot_acf(data['Monthly Mean Total Sunspot Number'])

#%% ARMA model
ARMA = ARIMA(data['Monthly Mean Total Sunspot Number'], order = (1,0,1)).fit()
print(ARMA.summary())

#%% ARMA visuallize
predict = ARMA.predict()
plt.plot(data['Monthly Mean Total Sunspot Number'])
plt.plot(predict, color ='red')

#%% Auto calculate pdq
pdq = auto_arima(data['Monthly Mean Total Sunspot Number'], trace= True, suppress_warnings=True)

#%% ARIMA model
ARIMA = ARIMA(data['Monthly Mean Total Sunspot Number'], order = (1,0,2)).fit()
print(ARIMA.summary())

#%% ARIMA visuallize
pre = ARIMA.predict()
plt.plot(data['Monthly Mean Total Sunspot Number'])
plt.plot(pre, color ='red')
# %%
