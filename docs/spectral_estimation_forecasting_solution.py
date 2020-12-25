'''
spectral analysis, estimation & forecasting
@Jimmy Azar
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#3-point moving average filter
#pandas
x = np.arange(1,10+1)
y = pd.Series(x).rolling(window=3, center=True).mean() #NaN will appear at ends

#alternatively: scipy
from scipy import signal
b = (np.ones(3))/3 
y = signal.convolve(x, b, mode='valid') #NaNs are automatically removed

x = np.random.normal(loc=0, scale=1, size=100)
y3 = pd.Series(x).rolling(window=3, center=True).mean()
y7 = pd.Series(x).rolling(window=7, center=True).mean()
#In this case the length of the filter should be odd, but if it is even, more of the filter is backward in time than forward.

plt.plot(x, label='original')
plt.plot(y3, lw=2, label='MA window=3')
plt.plot(y7, lw=2, label='MA window=7')
plt.title('Moving average filter')
plt.legend()
plt.show()

#example Lake Huron
path_to_file = './data/LakeHuron.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

y3 = data.rolling(window=3, center=True).mean()
y5 = data.rolling(window=5, center=True).mean()
y20 = data.rolling(window=20, center=True).mean()
y50 = data.rolling(window=50, center=True).mean()

t = data.index.values
plt.plot(data)
plt.plot(t, y3, t, y5, t, y20, t, y50, lw=2)
plt.xlabel('Time')
plt.ylabel('LakeHuron')
plt.legend(['original ts','3-point MA','5-point MA','20-point MA','50-point MA'])
plt.show()

#filter in series 
y2 = data.rolling(window=2, center=True).mean()
y2_repeated1 = y2.rolling(window=2, center=True).mean()
y2_repeated2 = y2_repeated1.rolling(window=2, center=True).mean()

plt.plot(data)
plt.plot(t, y2, t, y2_repeated1, t, y2_repeated2, lw=2)
plt.xlabel('Time')
plt.ylabel('LakeHuron')
plt.legend(['original ts','2-point MA','repeated once','repeated twice'])
plt.show()

#spencer filter
f = np.array([-3,-6,-5,3,21,46,67,74,67,46,21,3,-5,-6,-3])/320
y = signal.convolve(data.iloc[:,0].values, f, mode='valid') #NaNs are automatically removed

plt.plot(data, label='original ts')
plt.plot(t[7:-7], y, lw=2, label='Spencer filter') #note! remove ends of index 
plt.xlabel('Time')
plt.ylabel('LakeHuron')
plt.legend()
plt.show()

#differencing
path_to_file = './data/USAccDeaths.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

data.index = pd.to_datetime(data.index) 

#using shift()
#y = data - data.shift(periods=12) #gives NaNs for first elements

#using diff()
y = data.diff(periods=12) #same as above

#y.plot(legend=False) 
plt.plot(y)
plt.scatter(y.index.values, y, c='k')
plt.xlabel('Time')
plt.ylabel('Y')
plt.show()

#detrend the result 
w = y.diff(periods=1)

w.plot(legend=False) 
plt.scatter(w.index.values, w, c='k')
plt.axhline(y=0, c='r')
plt.xlabel('Time')
plt.ylabel('W')
plt.show()

#sinusoidal time series with noise
z = np.random.normal(loc=0, scale=1, size=100)

t = np.arange(0,100) 
x = 2*np.cos(2*np.pi/3*t) + z

plt.plot(t, x) 
plt.xlabel('Time')
plt.ylabel('X')
plt.show()

#a periodogram is the Fourier transform of the autocorrelation function
from scipy import signal

f, pdensity = signal.periodogram(x)  

plt.semilogy(f[1:], pdensity[1:]) #do not plot dc-value
plt.xlabel('frequency')
plt.ylabel('Spectrum')
plt.show()

#cumulative periodogram 
pdensity_cumulative = pdensity.cumsum()/pdensity.sum() #note! normalized into [0,1] by division

plt.plot(f, pdensity_cumulative) 
plt.plot([0, 0.5],[0, 1], c='gray', linestyle='--')
plt.xlabel('frequency')
plt.ylabel('Cumulative periodogram')
plt.show()

#simulation of AR(1)
z = np.random.normal(loc=0, scale=1, size=100)

x = [z[0]] #list
for i in range(1,100):
	x.append(1/3*x[i-1]+z[i])

plt.plot(x) #note x starts at 0 
plt.xlabel('Index')
plt.ylabel('X')
plt.show()

from statsmodels.tsa.stattools import acf, pacf, acovf
lag_acf = acf(x, nlags=20, fft=True) 
lag_pacf = pacf(x, nlags=20) 

#acf
plt.stem(lag_acf)
plt.axhline(y=-1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.show()

#pacf
plt.stem(lag_pacf)
plt.axhline(y=-1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(x)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('PACF')
plt.show()

#periodogram
f, pdensity = signal.periodogram(x)  

plt.semilogy(f[1:], pdensity[1:]) #do not plot dc-value
plt.xlabel('frequency')
plt.ylabel('Spectrum')
plt.show()

#cumulative periodogram
pdensity_cumulative = pdensity.cumsum()/pdensity.sum()

plt.plot(f, pdensity_cumulative) 
plt.plot([0, 0.5],[0, 1], c='gray', linestyle='--')
plt.xlabel('frequency')
plt.ylabel('Cumulative periodogram')
plt.show()

#simulation of a time series
from statsmodels.tsa.arima_process import ArmaProcess

model = ArmaProcess([1, -0.8897, 0.4858], [1, -0.2279, 0.2488]) 
ts0 = model.generate_sample(nsample=100)

plt.plot(ts0)
plt.title('ARMA(2,2)')
plt.show() 

model = ArmaProcess([1], [1, 0.8]) #MA(1)
ts1 = model.generate_sample(nsample=100)

model = ArmaProcess([1], [1, 0.01]) #MA(1)
ts2 = model.generate_sample(nsample=100)

plt.plot(ts1)
plt.plot(ts2)
plt.title('MA(1)')
plt.xlabel('Index')
plt.ylabel('Time series')
plt.legend(['theta=0.8','theta=0.01'])
plt.show() 

#estimation
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults 
#from statsmodels.tsa.arima_model import ARIMA  #same, but deprecated

model = ARIMA(ts1, order=(0, 0, 1)).fit()  
model.summary()

model = ARIMA(ts2, order=(0, 0, 1)).fit()  
model.summary()

model = ARIMA(ts0, order=(2, 0, 2)).fit()  
model.summary()

table_in_html = model.summary().tables[1].as_html()
df_summary = pd.read_html(table_in_html, header=0, index_col=0)[0]

#repeat estimation in a loop
phi = []
for i in range(1000):
	model = ArmaProcess([1, -0.8], [1]) #AR(1)
	x = model.generate_sample(nsample=100)
	
	model = ARIMA(x, order=(1, 0, 0)).fit()  
	table_in_html = model.summary().tables[1].as_html()
	df_summary = pd.read_html(table_in_html, header=0, index_col=0)[0]
	phi.append(df_summary.loc['ar.L1','coef'])
	
plt.hist(phi, density=False)
plt.grid()
plt.show()

np.mean(phi) #close to 0.8

#prediction
model = ArmaProcess([1], [1, 0.8]) #MA(1)
x = model.generate_sample(nsample=100)

model = ARIMA(x, order=(0, 0, 1)).fit()
y_pred = model.forecast(steps=2) 
 
#alternatively: same result
start = len(x)
end = len(x) + 1 #len(test) - 1
y_pred = model.predict(start=start, end=end, typ='levels')

model = ArmaProcess([1, -1.5, 0.75], [1]) 
x = model.generate_sample(nsample=50) + 30

X = pd.DataFrame(x, columns=['x'])
X_train, X_test = X.iloc[:40,:], X.iloc[40:,:]

model = ARIMA(X_train, order=(2, 0, 0)).fit()
y_pred = model.forecast(steps=10) 

table_in_html = model.summary().tables[1].as_html() 
df_summary = pd.read_html(table_in_html, header=0, index_col=0)[0]
sigma2 = df_summary.loc['sigma2','coef'] 
intercept = df_summary.loc['const','coef'] 

plt.plot(np.r_[X_train['x'].values, y_pred.iloc[0]], c='b', label='training')
plt.scatter(range(len(X_train)+1), np.r_[X_train['x'].values, y_pred.iloc[0]], c='b')
plt.plot(y_pred, linestyle='--', c='r', label='predicted')
plt.scatter(y_pred.index, y_pred, c='r')
plt.plot(y_pred.index, y_pred-1.96*np.sqrt(sigma2),linestyle='--',c='k')
plt.plot(y_pred.index, y_pred+1.96*np.sqrt(sigma2),linestyle='--',c='k')
plt.axhline(y=intercept,linestyle='--',c='gray')
plt.scatter(X_test.index, X_test, c='g' ,label='test')
plt.xlabel('Index')
plt.ylabel('Forecasting')
plt.legend()
plt.show()

pd.DataFrame(np.c_[X_test, y_pred], columns=['true','predicted'])

#lake Huron
path_to_file = './data/LakeHuron.csv'
data = pd.read_csv(path_to_file, index_col=0) 
data.head()

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
plt.show() 

#residual plots
residuals = y-y_pred

plt.plot(data.index.values, residuals) 
plt.scatter(data.index.values, residuals, c='k')
plt.axhline(y=0, c='k')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.show() 

lag_acf = acf(residuals, nlags=20, fft=True)
lag_pacf = pacf(residuals, nlags=20)

plt.stem(lag_acf)
plt.axhline(y=-1.96/np.sqrt(len(residuals)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(residuals)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.xticks(np.arange(0,20+1,2))
plt.show() 

plt.stem(lag_pacf)
plt.axhline(y=-1.96/np.sqrt(len(residuals)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(residuals)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('PACF')
plt.xticks(np.arange(0,20+1,2))
plt.show() 

#fit an ARMA(1,3) model
model = ARIMA(residuals, order=(1, 0, 3)).fit()
model.summary()

#diagnostic checking
plt.plot(model.resid) #note! model.resid necessary to access rest below 
plt.xlabel('Index')
plt.ylabel('Model residuals')
plt.show() 

#cumulative periodogram 
f, pdensity = signal.periodogram(model.resid)  
pdensity_cumulative = pdensity.cumsum()/pdensity.sum() #note! normalized into [0,1] by division

plt.plot(f, pdensity_cumulative) 
plt.plot([0, 0.5],[0, 1], c='gray', linestyle='--')
plt.xlabel('frequency')
plt.ylabel('Cumulative periodogram')
plt.show()

lag_acf = acf(model.resid, nlags=20, fft=True)
#lag_pacf = pacf(model.resid, nlags=20)

plt.stem(lag_acf)
plt.axhline(y=-1.96/np.sqrt(len(model.resid)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(model.resid)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('ACF')
plt.xticks(np.arange(0,20+1,2))
plt.show() 

'''
plt.stem(lag_pacf)
plt.axhline(y=-1.96/np.sqrt(len(model.resid)),linestyle='--',c='gray')
plt.axhline(y=1.96/np.sqrt(len(model.resid)),linestyle='--',c='gray')
plt.xlabel('lag')
plt.ylabel('PACF')
plt.xticks(np.arange(0,20+1,2))
plt.show() 
'''

plt.hist(model.resid, density=False)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()

from scipy import stats
stats.probplot(model.resid, dist='norm', plot=plt) #need to set plot argument to show the plot
plt.show()

stat, pvalue = stats.shapiro(model.resid)
print(f'stat={stat}, pvalue={pvalue}')

#auto_arima()
'''
from pmdarima import auto_arima
stepwise_fit = auto_arima(df['AvgTemp'], trace=True, suppress_warnings=True)

import pmdarima as pmd

def arimamodel(timeseriesarray):
    autoarima_model = pmd.auto_arima(timeseriesarray, 
                              start_p=1, 
                              start_q=1,
                              test="adf",
                              trace=True)
    return autoarima_model

arima_model = arimamodel(train_array)
arima_model.summary()
'''
