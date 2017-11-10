# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:00:22 2017

@author: masantana
"""

#Import libraries 

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
from pandas import DataFrame

warnings.filterwarnings("ignore")
pvalue = []

#Read the csv file and get only the time series

series = pd.read_csv('C:\Git\sdl\code\dados\lag15Min-PETR4.csv')
series.drop('Unnamed: 0', axis = 1, inplace = True)
series.drop('INSTRUMENT', axis = 1, inplace = True)
series.drop('TRADE_TIME', axis = 1, inplace = True)
print(series.head())
plt.plot(series)
plt.title('PETR4 May fluctuation - lag 15min')
plt.xlabel('Time (min)')
plt.ylabel('Price (Brazilian Reais)')
plt.show()

#Plot correlation and autocorrelation plots to analyze how the numbers relate

print ('ACF and PACF for the original series')
pyplot.figure()
plot_acf(series, ax=pyplot.gca(), lags = 20, title = 'Autocorrelation plot for original series')
pyplot.show()
pyplot.figure()
plot_pacf(series, ax=pyplot.gca(), lags = 20, title = 'Partial Autocorrelation plot for original series')
pyplot.show()

#Function to plot the mean&variation graph 

def plotstats(timeseries):
    
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:

    plt.figure(1)
    plt.plot(timeseries, color = 'blue', label = 'Original')
    plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    plt.plot (rolstd, color = 'black', label = 'Rolling Std')
    plt.legend (loc = 'best')
    plt.title ('Rolling Mean & Standand Deviation for PETR4')
    plt.show (block = False)
    
#Function to test stationarity using the Dicker-Fuller Test
    
def test_stationarity(timeseries):

    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags used', 'Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return dfoutput[1]
    
print ('Lets try with 1 order of differencing')

order_of_diferencing = 1
differencing = series - series.shift(periods=order_of_diferencing)
differencing.dropna(inplace=True)

#Plot the differenced series to check stationarity

plt.figure(5)
plotstats (series.values)
plt.show()

plt.figure(6)
plt.plot(differencing)
plotstats (differencing)
plt.show()

print ('ACF and PACF with series stationarized')

pyplot.figure()
plot_acf(differencing, ax=pyplot.gca(), lags = 20)
pyplot.figure()
plot_pacf(differencing, ax=pyplot.gca(), lags = 20)
pyplot.show()

lag_acf = acf (differencing, nlags = 20)
lag_pacf = pacf (differencing, nlags = 20, method = 'ols')

#Temporary test ACF and PACF

plt.figure(13)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle = '--', color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(series)), linestyle = '--', color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(series)), linestyle = '--', color = 'gray')
plt.title ('Autocorrelation function for PETR4 - ARIMA (0,1,1)')
plt.show()

#Plot PACF:
    
plt.figure(14)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle = '--', color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(series)), linestyle = '--', color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(series)), linestyle = '--', color = 'gray')
plt.title ('Partial Autocorrelation function for PETR4 - ARIMA (0,1,1)')
plt.show()

#Add pvalues to mathematically analyze if the series is stationary or not

pvalue.append (test_stationarity(series['TRADE_PRICE']))
pvalue.append (test_stationarity(log(series['TRADE_PRICE'])))
pvalue.append (test_stationarity(differencing['TRADE_PRICE']))

#Apply ARIMA:
    
#First we have to divite between dataset and validation (train and test)
#Here we are using 65% for train and 35% for test

initial_serie = series.values
split_point = int(0.65*len(series))
train = initial_serie[0:split_point]
test = initial_serie[split_point:]

#ARIMA parameters

order_of_differencing = 1
p_value=0
q_value=1

history = [x for x in train]

predictions = list()

for t in range (len(test)):
    model = ARIMA (history, order = (p_value,order_of_differencing,q_value))
    model_fit = model.fit(disp=-1)
    output = model_fit.forecast()[0]
    yhat = output[0]
    predictions.append(yhat)
    history.append(test[t])
    print('predicted=%f, expected=%f' %(yhat,test[t]))

#Test up and down 

counter = list()

for s in range (1,5):

    up_pred = [False]*int((len(test)-s))
    down_pred = [False]*int((len(test)-s))
    up_test = [False]*int((len(test)-s))
    down_test = [False]*int((len(test)-s))
    temp = 0
    
    for t in range (s,int((len(test)-s-1))):
        if (predictions[t]>predictions[t-s]):
            up_pred[t-s] = True 
        else: down_pred[t-s] = True
    
        if (test[t]>test[t-s]):
            up_test[t-s] = True
        else: down_test[t-s] = True
    
        if (up_pred[t-s] == True and up_test[t-s] == True) or (down_pred[t-s] == True and down_test[t-s] == True):
            temp+=1
            
    counter.append(temp/len(up_pred))
        
#Test error in cents

error_final = list()

for t in range (0,int(len(test)-1)):
    error_final.append(abs(float(predictions[t]-test[t])))
    
results = DataFrame()
results['ERROR'] = error_final
print(results.describe())
results.boxplot()
pyplot.title('Error PETR4')
pyplot.show()
    
#Calculate RMSE

error = mean_squared_error(test, predictions)
print('Test RMSE: %.8f'%sqrt(error))
array = (abs(100*(test[t]-predictions[t])/test[t]) for t in range(len(test)))
array = DataFrame(array)
average = np.mean(array)

#Plot predictions vs test

plt.figure(10)
pyplot.plot(test, color = 'blue', label = 'Real values')
pyplot.plot(predictions, color = 'red', label = 'Predictions')
pyplot.title('PETR4 predictions')
pyplot.legend (loc = 'best')
pyplot.show()

#It is important to plot the ACF and PACF for the residuals to make sure we chose the right parameters

print ('ACF and PACF for the residuals')
residuals_2 = ((test[t] - predictions[t]) for t in range(len(test)))
residuals_2 = DataFrame (residuals_2)
pyplot.figure()
plot_acf(residuals_2, ax=pyplot.gca(), lags = 20, title = 'Autocorrelation plot for residuals')
pyplot.show()
pyplot.figure()
plot_pacf(residuals_2, ax=pyplot.gca(), lags = 20, title = 'Partial Autocorrelation plot for residuals')
pyplot.show()



