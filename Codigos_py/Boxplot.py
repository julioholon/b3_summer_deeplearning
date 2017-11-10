# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:06:33 2017

@author: masantana
"""

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
from pylab import boxplot

warnings.filterwarnings("ignore")
pvalue = []

#Read the csv file and get only the time series

series = pd.read_csv('C:\Git\sdl\code\dados\lag15Min-BVMF3.csv')
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
p_value=1
q_value=0

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
results['ARIMA'] = error_final
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
pyplot.title('ITUB4 predictions')
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

########################################################################

#LSTM 

#Import libraries

from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np
import pandas as pd

#Batch_size must be set to 1 because we are interested in making one step forecasts on the test data and this is constrained by the predict() function
batch_size = 1 

#Inputs we can change
num_epochs = 50
layer_size = 3
timesteps = 1
input_dim = 1
train_test_split = 0.65
times = 10

errors_test = list()
errors_train = list()
counter=list()

#Read the csv file and get only the time series

series = pd.read_csv('C:\Git\sdl\code\dados\lag15Min-BVMF3.csv')
series.drop('Unnamed: 0', axis = 1, inplace = True)
series.drop('INSTRUMENT', axis = 1, inplace = True)
series.drop('TRADE_TIME', axis = 1, inplace = True)

#Frame the sequence as a supervized learning problem

def timeseries_to_supervised(data, lag=1):
    df=DataFrame(data)
    columns = [df.shift(i) for i in reversed(range(1,lag+1))]
    columns.append(df)
    df = concat(columns, axis = 1)
    df.fillna(0, inplace=True) 
    return df
   
#Stationarize the series

def difference (dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value[0])
    return Series(diff)

#Function to inverse the difference for a forecasted value

def inverse_difference(history, yhat, interval):
    return yhat + history[-interval]

#Scale train and test data to [-1,1] - because of the hiperbolic tangent function inside the LSTM neuron

def scale (train, test):
    scaler = MinMaxScaler (feature_range = (-1,1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#Inverse scaling for a forecasted value

def invert_scale (scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1,len(array))
    inverted = scaler.inverse_transform(array)
    return inverted [0,-1]

#LSTM network for training data

def fit_lstm (train, batch_size, nb_epoch, neurons, timesteps=1, input_dim=1):
    X, y = train[:, 0:-1], train[:, -1]
    
    #The input has to be in the form X(nb_samples, timesteps, input_dim(features)), so we have to reshape X
    #Features = imput_dim = separate measures observed at the time of observation
    
    X = X.reshape(X.shape[0], timesteps, input_dim)
    model = Sequential()
    
    #The shape of the input data must also be specified in the LSTM layer using the batch_input_shape
    #batch_input_shape = (nb_samples, timesteps, input_dim)
    
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful = True, return_sequences=True))
    
    #In order to put more than one hidden layer return_squences = True
    
    model.add(LSTM(neurons,batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful = True))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    for i in range (nb_epoch):
        model.fit(X, y, epochs=1, batch_size = batch_size, verbose = 0, shuffle = False)
        model.reset_states()
    return model

#Make a one-step forecast

def forecast_lstm (model, batch_size, X):
    X = X.reshape(1, timesteps, input_dim)
    yhat = model.predict(X, batch_size = batch_size)
    return yhat[0,0]
    
#Transform data to be stationary

raw_values = series.values
diff_values = difference(raw_values, 1)

#Tranform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, timesteps)
supervised_values = supervised.values

#Split data into train and test-sets
train, test = supervised_values[0:int(train_test_split*len(supervised_values))], supervised_values[int(train_test_split*len(supervised_values)):]

#Transform the scale 
scaler, train_scaled, test_scaled = scale(train, test)

error = [0]*int(len(test))

for t in range(times):
    
    #Fit the model
    lstm_model = fit_lstm(train_scaled,batch_size,num_epochs,layer_size,timesteps,input_dim)
    
    #Forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:,:-1].reshape(train_scaled.shape[0],timesteps,input_dim)
    lstm_model.predict(train_reshaped, batch_size)    
        
    #Make lists for train predictions and test predictions to compare and see how we can optimize the algorithm
    
    predictions_train = list()
    predictions_test = list ()
    
    for i in range(len(train_scaled)):
        #Read one line
        X2, y2 = train_scaled[i,0:-1], train_scaled[i,-1]
        yhat2 = forecast_lstm(lstm_model,batch_size,X2)
        #invert scaling
        yhat2 = invert_scale(scaler, X2, yhat2)
        #invert differencing
        yhat2 = inverse_difference(raw_values, yhat2, -i)
        #store forecast
        predictions_train.append(yhat2)
        expected2 = raw_values[i]
        print('Predicted = %f, Expected = %f' %(yhat2, expected2))
    
    for i in range(len(test_scaled)):
        #Read one line
        X, y = test_scaled[i,0:-1], test_scaled[i,-1]
        yhat = forecast_lstm(lstm_model,batch_size,X)
        #invert scaling
        yhat = invert_scale(scaler,X, yhat)
        #invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        #store forecast
        predictions_test.append(yhat)
        expected = raw_values[len(train)+i+1]
        print('Predicted = %f, Expected = %f' %(yhat, expected))
        
    #RMSE for train predictions and test predictions
    
    rmse = sqrt(mean_squared_error(raw_values[-len(predictions_test):], predictions_test))
    errors_test.append(rmse)
    print ('Test RMSE: %.8f' %rmse)
    
    
    rmse2 = sqrt(mean_squared_error(raw_values[0:len(predictions_train)], predictions_train))
    errors_train.append(rmse2)
    print ('Train RMSE: %.8f' %rmse2)
    
    #Errors in cents
    
    temp2 = list()
  
    for s in range (0,int(len(predictions_test))):
        temp2.append(abs(predictions_test[s] - raw_values[len(raw_values)-len(predictions_test)+s]))
    
    for s in range (0,int(len(predictions_test))):
        error[s] = (np.float(error[s]) + np.float(temp2[s]))/2
    
#Test up and down
    
    columns2 = list()
    
    for s in range (1,5):
        
        temp = 0
        up_pred = [False]*int((len(test)-s))
        down_pred = [False]*int((len(test)-s))
        up_test = [False]*int((len(test)-s))
        down_test = [False]*int((len(test)-s))
        
        for t in range (s,int((len(test)-1-s))):
            if (predictions_test[t]>predictions_test[t-s]):
                up_pred[t-s] = True 
            else: down_pred[t-s] = True
        
            if (raw_values[len(raw_values)-len(predictions_test)+t]>raw_values[len(raw_values)-len(predictions_test)-s+t]):
                up_test[t-s] = True
            else: down_test[t-s] = True
        
            if (up_pred[t-s] == True and up_test[t-s] == True) or (down_pred[t-s] == True and down_test[t-s] == True):
                temp+=1

        columns2.append(np.array(temp/len(up_pred))) 
        
    counter.append(columns2)
    
#Loop to calculate the average of up-down accuracy 

final_mean = list()
sum = 0
 
for i in range(0,4):
    sum = 0
    for j in range (0,times):
        sum = sum + np.float(counter[j][i])
        mean = sum/times
    final_mean.append(mean)
    
#Observation of results
    
#results = DataFrame()
#results['RMSE Test'] = errors_test
#print(results.describe())
#results.boxplot()
#pyplot.show()
#
#results2 = DataFrame()
#results2['RMSE Train'] = errors_train
#print(results2.describe())
#results2.boxplot()
#pyplot.show()

results3 = DataFrame()
results3['LSTM'] = error
print(results3.describe())
new = pd.concat([results, results3], axis = 1)
new.boxplot()
pyplot.title('Simple Errors BVMF3')
pyplot.show()

#line plot of observed vs predicted

pyplot.plot(raw_values[-len(predictions_test):])
pyplot.plot(predictions_test)
pyplot.title('LSTM Predictions PETR4')
pyplot.show()