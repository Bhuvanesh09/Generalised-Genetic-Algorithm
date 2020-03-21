# Importing Libraries 
import numpy as np 
import pandas as pd 
import sklearn 
# Importing Data 
from sklearn.datasets import load_boston 
boston = load_boston() 
data = pd.DataFrame(boston.data) 
data.columns = boston.feature_names 
data['Price'] = boston.target 

x = boston.data
x = x[:,:10]
y = boston.target 

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2, random_state = 0) 
def get_error(*w):
	err = []
	for j in range(404):
	  y_pred = 0
	  for i in range(10):

	    y_pred += w[i]*x[j][i]
	  err.append(y[j]-y_pred)
	mse = 0
	for i in range(404):
	  mse += err[i]*err[i]
	return mse
	 
