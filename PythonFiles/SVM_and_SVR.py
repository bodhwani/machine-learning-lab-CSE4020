
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # Classification

# In[6]:

import numpy as np
import urllib
# url with dataset
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# # download the file
# raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = pd.read_csv('mtcars.csv')
# separate the data from the target attributes
# X = dataset[:,0:8]
# y = dataset[:,8]
# print X
# print y
print dataset.head()


# In[10]:

y = dataset['am']
X = dataset.drop(['am','model'],axis=1)
print "This is the dataset size - training examples",y.shape


# In[54]:

print "No of testing examples",y.shape


# In[11]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# # Defining classifier using SVC from SVM classifier

# In[12]:

clf = SVC(kernel = 'linear', C=1000)


# # Training

# In[13]:

clf.fit(X_train, y_train)


# In[14]:

pred = clf.predict(X_test)


# # Output :
# ### Prediction class

# In[15]:

print pred


# # Accuracy :

# In[16]:

accuracy = accuracy_score(pred, y_test)
print("Here is the new accuracy",accuracy)


# In[17]:

get_ipython().magic(u'matplotlib inline')
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');


# # Regression :

# In[76]:

from sklearn.svm import SVR
dataset = pd.read_csv('mtcars.csv')
print dataset.head


# In[77]:

X = dataset.as_matrix(['mpg','drat']).astype('float32')
y = dataset.as_matrix(['wt'])


# # Graph

# In[81]:

get_ipython().magic(u'matplotlib inline')
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');


# # Training

# In[82]:

X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2)


# # SVM using SVR regressor

# In[83]:

clfr = SVR(C=1.0, epsilon=0.2)


# In[84]:

clfr.fit(X_train, y_train)


# # Accuracy :
# As seen from the graph above, because of dataset we are getting less accuracy

# In[86]:

score = clfr.score(X_test,y_test)
print "Score is ",score

