
# coding: utf-8

# In[37]:

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn import preprocessing,cross_validation,neighbors
from sklearn.metrics import accuracy_score


# In[38]:

dataset = pd.read_csv('dataset3.csv')
print dataset
print dataset.dtypes


# # Scatter Plots

# In[39]:

brain = dataset['Brain']
piq = dataset['PIQ']
sns.regplot(x='Brain', y='PIQ', data=dataset, fit_reg=False)
plt.show()


# In[55]:

x_train, x_test, y_train,y_test= cross_validation.train_test_split(dataset['Brain'],dataset['PIQ'],test_size=0.2)


# In[56]:

reg = linear_model.LinearRegression()
reg.fit(x_train.values.reshape(-1,1), y_train.values.reshape(-1,1))
print reg.score(x_test.values.reshape(-1,1),y_test.values.reshape(-1,1))


# In[57]:

x_line = np.arange(80,100).reshape(-1,1)
sns.regplot(x=dataset['Brain'], y=dataset['PIQ'], data=dataset, fit_reg=False)
plt.plot(x_line, reg.predict(x_line))
plt.show()

