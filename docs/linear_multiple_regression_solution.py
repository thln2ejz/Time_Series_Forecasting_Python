'''
linear & multiple regression
@Jimmy Azar
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

path_to_file = './data/mtcars.csv'

data = pd.read_csv(path_to_file)
data.head()

plt.scatter(data['wt'], data['mpg'])
plt.xlabel('Weight')
plt.ylabel('Miles per gallon')
plt.show() 

from sklearn.linear_model import LinearRegression

X, y = data['wt'], data['mpg']
model = LinearRegression().fit(X.values.reshape(-1,1), y) #direct use of reshape() deprecated in pandas
model.coef_ 
model.intercept_ 
#model.score(X.values.reshape(-1,1), y) #coefficient of determination R^2
 
X_test = np.linspace(X.min(),X.max(),50)
y_pred = model.predict(X_test.reshape(-1,1))

plt.scatter(data['wt'], data['mpg'])
plt.plot(X_test, y_pred, c='r', lw=2)
plt.xlabel('Weight')
plt.ylabel('Miles per gallon')
plt.show() 

y_pred = model.predict(X.values.reshape(-1,1))
residuals = y - y_pred  #residuals over training set

plt.scatter(residuals.index,residuals) 
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show() 

plt.hist(residuals, density=False)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of residuals')
plt.grid()
plt.show()

#qqplot using statsmodels (however shapiro test is only available in scipy)
'''
import statsmodels.api as sm
#sm.qqplot(residuals, line='r')
#sm.qqplot(residuals, line='s')
sm.qqplot(residuals, line='q') #same as R's
plt.show()
'''

#alternatively:
from scipy import stats

stats.probplot(residuals, dist='norm', plot=plt) #need to set plot argument to show the plot
plt.show()

stat, pvalue = stats.shapiro(residuals)
print(f'stat={stat}, pvalue={pvalue}')

#Cook's distance:
from statsmodels.formula.api import ols

model = ols('mpg ~ wt',data).fit() #re-fitting using statsmodels
influence = model.get_influence()
influence.summary_frame()
cooks_distance = influence.summary_frame()['cooks_d']

plt.scatter(cooks_distance.index, cooks_distance)
plt.xlabel('Index')
plt.ylabel('Cook\'s distance')
plt.show()

#predictions
predictions = model.get_prediction(pd.DataFrame(X_test, columns=['wt']))
df = predictions.summary_frame(alpha=0.05) #alpha is the significance level
y_pred = df['mean']
y_lower = df['obs_ci_lower'] #prediction intervals
y_upper = df['obs_ci_upper']

plt.scatter(X, y)
plt.plot(X_test, y_pred, c='r', lw=2)
plt.plot(X_test, y_lower, 'r--', lw=2)
plt.plot(X_test, y_upper, 'r--', lw=2)
plt.xlabel('wt')
plt.ylabel('mpg')
plt.show()

#multiple regression (use sklearn again)
X = data[['wt','hp']]
model = LinearRegression().fit(X, y)
model.coef_ 
model.intercept_ 

y_pred = model.predict(np.array([[3.5, 90]]))
y_pred

#alternatively: using ols's model
model = ols('mpg ~ wt + hp',data).fit() 
model.predict(pd.DataFrame([[3.5, 90]],columns=['wt','hp']))
