
# coding: utf-8

# # Logistic Regression

# In[26]:

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import preprocessing,cross_validation,neighbors
from sklearn.metrics import accuracy_score


# In[27]:

dataset = pd.read_csv('dataset3.csv')


# In[28]:

print dataset.head()
print dataset.dtypes


# In[53]:

X = dataset.as_matrix(['Brain','Weight']).astype('float32')
Y = dataset.as_matrix(['PIQ'])


x_total = X
y_total = Y

x_train, x_test, y_train,y_test= cross_validation.train_test_split(x_total,y_total,test_size=0.1)


# In[54]:

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print y_pred
print('Score: ', reg.score(x_test, y_test))



# In[ ]:



