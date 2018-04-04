
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import preprocessing,cross_validation,neighbors
from sklearn.metrics import accuracy_score


# In[3]:

dataset = pd.read_csv('dataset3.csv')


# In[4]:

print dataset.head()
print dataset.dtypes


# In[5]:

brain = dataset['Brain']
weight = dataset['Weight']
height = dataset['Height']

piq = dataset['PIQ']


x_total = []

# print brain
#print piq
# y_total = [x for x in piq]
# for i in brain:
#     for j in weight:
#         x = x.append(i,j)
#         break
#     x_total.append(x)
brainx=  [y for y in brain]
weightx = [x for x in weight]
heightx = [z for z in height]
a=zip(brainx, weightx, heightx)
for i in range(len(a)):
    a[i]=list(a[i])

y_total = [x for x in piq]
x_total = a

print x_total





# In[7]:



x_train, x_test, y_train,y_test= cross_validation.train_test_split(x_total,y_total,test_size=0.2)


# In[8]:

reg = linear_model.LinearRegression()


# In[11]:

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print y_pred


print('Coefficients: \n', reg.coef_)
# The mean squared error



# In[12]:

train_color = "b"
test_color = "r"
score = reg.score(x_train,y_train)
print "Here is the score on training data",score




# In[ ]:



