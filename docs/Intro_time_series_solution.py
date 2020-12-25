'''
Introduction to time series analysis
@Jimmy Azar
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#plot time series 
x = np.random.normal(loc=3, scale=2, size=100) 

plt.scatter(range(len(x)), x)
plt.plot(x, linestyle='dashed', label='N(3,2)')
plt.plot([0,len(x)], [3,3], lw=2, label='mean')
plt.scatter(50,5,marker='+',c='k',lw=2)
plt.xlabel('Index')
plt.ylabel('x')
plt.legend()
plt.show()

#white noise time series 
x = np.random.normal(loc=0, scale=1, size=200) 

xt = pd.Series(x, index=range(1800,2000))

plt.plot(xt)
plt.xlabel('Year')
plt.ylabel('x')
plt.title('White noise')
plt.show()

#time series datasets (taken from built-in R datasets)

#data: number of accidental deaths in the US 1973-1978
path_to_file = './data/USAccDeaths.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

data.index = pd.to_datetime(data.index) 
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('USAccDeaths')
plt.show()

#data: annual measurements of the level, in feet, of Lake Huron 1875-1972.
path_to_file = './data/LakeHuron.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

plt.plot(data)
plt.xlabel('Time')
plt.ylabel('LakeHuron')
plt.show()

#data: monthly Airline Passenger Numbers 1949-1960
path_to_file = './data/AirPassengers.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

data.index = pd.to_datetime(data.index)
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('AirPassengers')
plt.show()

#data: annual numbers of lynx trappings for 1821-1934 in Canada.
path_to_file = './data/lynx.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

plt.plot(data)
plt.xlabel('Time')
plt.ylabel('lynx')
plt.show()

#trend estimation by linear regression
path_to_file = './data/LakeHuron.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

plt.plot(data)
plt.scatter(data.index, data, marker='o', facecolors='none', edgecolors='k')
plt.xlabel('Time')
plt.ylabel('LakeHuron')
plt.show()

from sklearn.linear_model import LinearRegression

X, y = data.index.values, data.values
model = LinearRegression().fit(X.reshape(-1,1), y)
model.coef_ 
model.intercept_ 
y_pred = model.predict(X.reshape(-1,1))

plt.plot(data) 
plt.scatter(data.index, data, marker='o', facecolors='none', edgecolors='k')
plt.plot(X, y_pred, c='r', lw=2)
plt.xlabel('Year')
plt.ylabel('LakeHuron')
plt.show() #Figure 1-9

#residual plots
residuals = y-y_pred

plt.plot(data.index, residuals) 
plt.scatter(data.index, residuals, c='k')
plt.axhline(y=0, c='k')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.show() #Figure 1-10

#harmonic regression
#Example 1.3.6 Accidental death
path_to_file = './data/USAccDeaths.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

data.index = pd.to_datetime(data.index)
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('USAccDeaths')
plt.show()

t = np.arange(0,len(data))  #starts at 0, so coef_ will differ slightly from R's
x1 = np.cos(6*2*np.pi/72*t)
x2 = np.sin(6*2*np.pi/72*t)
x3 = np.cos(12*2*np.pi/72*t)
x4 = np.sin(12*2*np.pi/72*t)

X, y = np.c_[x1,x2,x3,x4], data.values
model = LinearRegression().fit(X, y)
model.coef_ 
model.intercept_ 
y_pred = model.predict(X)

plt.plot(data)
plt.scatter(data.index.values, data, c='k')
plt.plot(data.index.values, y_pred, c='r', lw=2)
plt.xlabel('Time')
plt.ylabel('USAccDeaths')
plt.show() #Figure 1-11

#AR(1) assumption
#Figure 1-16 
from statsmodels.tsa.stattools import acf, pacf, acovf

lag_acf = acf(residuals, nlags=20, fft=True)

plt.stem(lag_acf)
#plt.axhline(y=0, linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(residuals)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(residuals)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.show() 

#alternatively: 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(residuals, lags=20)
plt.show()

X, y = residuals[:-1], residuals[1:]

model = LinearRegression().fit(X, y)
model.coef_ 
model.intercept_ 
y_pred = model.predict(X)

phi = model.coef_.item() #np.asscalar(model.coef_) is deprecated 
print(phi) 

plt.scatter(X, y)
plt.plot(X, y_pred, c='r', lw=2)
plt.xlabel('y(t-1)')
plt.ylabel('y(t)')
plt.show() #Figure 1-16

#ACVF and ACF 
path_to_file = './data/USAccDeaths.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

data.index = pd.to_datetime(data.index)
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('USAccDeaths')
plt.show()

lag_acf = acf(data, nlags=20, fft=True)
lag_acovf = acovf(data, nlag=20, fft=True)

plt.stem(lag_acf)
#plt.axhline(y=0, linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.show() 

#alternatively: 
#plot_acf(data, lags=20)
#plt.show()

plt.stem(lag_acovf)
#plt.axhline(y=0, linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACVF')
plt.show() 

#note! plot_acovf() does not exist in python 

#white noise: Example 1.4.6
x = np.random.normal(loc=0, scale=1, size=200)

lag_acf = acf(x, nlags=20, fft=True) 

plt.stem(lag_acf)
#plt.axhline(y=0, linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('white noise')
plt.show() #Figure 1-13

#theoretical ACF versus sample ACF
x = np.random.normal(loc=0, scale=2, size=100)
lag_acf = acf(x, nlags=20) 

plt.stem(lag_acf, markerfmt=' ')
#plt.axhline(y=0, linestyle='--',c='gray')
plt.axhline(y=-1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.scatter(range(20+1),np.hstack([1,np.zeros(20)]),c='r')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('white noise')
plt.xticks(np.arange(0,20+1,2))
plt.show() 

#MA(1) process
acf_theoretical = [1, 0.8/(1+0.8*0.8)] + [0]*19  #appending 2 lists

plt.stem(acf_theoretical)
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('MA(1)')
plt.xticks(np.arange(0,21+1,2))
plt.show() 

#simulate an MA(1)-process with theta=0.8
from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess

ar_params = np.array([1])
#ar_params = np.r_[1, -ar_params] #add zero-lag and negate 
ma_params = np.array([.8])
ma_params = np.r_[1, ma_params] # add zero-lag

model = ArmaProcess(ar_params, ma_params)
model.isstationary 
model.isinvertible
model.arroots
y_simulated = model.generate_sample(nsample=100)

plt.plot(y_simulated)
plt.title('MA(1)')
plt.show() 

lag_acf = acf(y_simulated, nlags=20) 

plt.stem(lag_acf, markerfmt=' ')
plt.axhline(y=-1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.scatter(range(20+1),acf_theoretical,c='r')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('MA(1)')
plt.xticks(np.arange(0,21+1,2))
plt.show() 

#AR(1) process
acf_theoretical = (-0.5)**np.arange(20+1) 

plt.stem(acf_theoretical)
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('AR(1)')
plt.xticks(np.arange(0,21+1,2))
plt.show() 

#simulate an AR(1)-process with phi=-0.5
ar_params = np.array([-0.5])
ar_params = np.r_[1, -ar_params] #add zero-lag and negate 
ma_params = np.array([1])
#ma_params = np.r_[1, ma_params] # add zero-lag

model = ArmaProcess(ar_params, ma_params) 
model.isstationary 
model.isinvertible
model.arroots
y_simulated = model.generate_sample(nsample=100)

plt.plot(y_simulated)
plt.title('AR(1)')
plt.show() 

lag_acf = acf(y_simulated, nlags=20) 

plt.stem(lag_acf, markerfmt=' ')
plt.axhline(y=-1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.scatter(range(20+1),acf_theoretical,c='r')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('AR(1)')
plt.xticks(np.arange(0,21+1,2))
plt.show() 

#alternatively: use method .acf() 
acf_theoretical = (-0.5)**np.arange(20+1)
acf_theoretical = model.acf(lags=20+1) #same result

#ARMA(2,2) process
ar_params = np.array([-0.3, 0.1])
ar_params = np.r_[1, -ar_params] #add zero-lag and negate 
ma_params = np.array([0.5, 0.8])
ma_params = np.r_[1, ma_params] # add zero-lag

model = ArmaProcess(ar_params, ma_params) 
model.isstationary 
model.isinvertible
model.arroots
y_simulated = model.generate_sample(nsample=100)

plt.plot(y_simulated)
plt.title('ARMA(2,2)')
plt.show() 

lag_acf = acf(y_simulated, nlags=20) 
acf_theoretical = model.acf(lags=20+1)

plt.stem(lag_acf, markerfmt=' ')
plt.axhline(y=-1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.scatter(range(20+1),acf_theoretical,c='r')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.title('ARMA(2,2)')
plt.xticks(np.arange(0,21+1,2))
plt.show() 

#partial autocorrelation function (PACF)
#AR(3) process
ar_params = np.array([0.5, -0.2, 0.1])
ar_params = np.r_[1, -ar_params] #add zero-lag and negate 
ma_params = np.array([1])
#ma_params = np.r_[1, ma_params] # add zero-lag

model = ArmaProcess(ar_params, ma_params) 
model.isstationary 
model.isinvertible
model.arroots
y_simulated = model.generate_sample(nsample=100)

#plt.plot(y_simulated)
#plt.title('AR(3)')
#plt.show() 

lag_pacf = pacf(y_simulated, nlags=20)   #sample PACF
pacf_theoretical = model.pacf(lags=20+1) #theoretical PACF ("not test/checked yet" in documents!)

plt.stem(lag_pacf, markerfmt=' ')
plt.axhline(y=-1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(y_simulated)),linestyle='--',c='gray')
plt.scatter(range(20+1),pacf_theoretical,c='r')
plt.xlabel('lag')
plt.ylabel('PACF')
plt.title('AR(3)')
plt.xticks(np.arange(0,21+1,2))
plt.show() 

#plot_pacf(y_simulated)
#plt.show() 
