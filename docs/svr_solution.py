'''
support vector regression
@Jimmy Azar
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Exercise
def synth_data(n_samples, sigma):
	X = np.random.uniform(low=0,high=2,size=n_samples)
	X = np.sort(X)
	y = np.sin(4*X) + np.random.normal(loc=0,scale=sigma,size=n_samples) 
	return X, y
	
X_train, y_train = synth_data(n_samples=50, sigma=0.1)

plt.scatter(X_train, y_train)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

from sklearn.svm import SVR
model = SVR(kernel='rbf', C=1, epsilon=0.1, gamma='scale').fit(X_train.reshape(-1,1), y_train) 
y_pred = model.predict(X_train.reshape(-1,1))

plt.scatter(X_train, y_train, c='black', label='original')
plt.scatter(X_train, y_pred, c='red', marker='x', lw=2, label='SVR predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show() 

rmse = np.sqrt(((y_train - y_pred)**2).mean())

#Exercise
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ['rbf'], 
			  'gamma': np.linspace(1,10,20),
			  'epsilon': np.linspace(0,1,10),
			  'C': 2**np.arange(0,10)}
  
model = GridSearchCV(SVR(), parameters, scoring='neg_mean_squared_error', cv=5) #neg_mean_squared_error is used because the method selects model with max criterion by default
model.fit(X_train.reshape(-1,1), y_train)
model.best_params_
#model.best_estimator_
#model.best_score_
y_pred = model.predict(X_train.reshape(-1,1)) 

rmse = np.sqrt(((y_train - y_pred)**2).mean()) #less than before (0.07<0.09)

plt.scatter(X_train, y_train, c='black', label='original')
plt.scatter(X_train, y_pred, c='red', marker='x', lw=2, label='SVR predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show() 

#Exercise
X_test = np.arange(-1,3,0.05)
y_pred = model.predict(X_test.reshape(-1,1))

plt.scatter(X_train, y_train, c='black', label='original')
plt.scatter(X_test, y_pred, c='red', marker='x', lw=2, label='SVR predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show() 
