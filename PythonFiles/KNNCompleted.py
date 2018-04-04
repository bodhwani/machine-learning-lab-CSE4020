
# coding: utf-8

# In[23]:

import pandas as pd
import numpy as np
from sklearn import neighbors
import sklearn
from sklearn import preprocessing,cross_validation,neighbors



# In[24]:

dataset = pd.read_csv('knndataset.csv')
dataset.head()


# In[25]:

X = dataset.as_matrix(['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'])
Y = dataset.as_matrix(['Species'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.4, random_state=0)


# In[26]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print knn.predict(X_test)
print('Score: ', knn.score(X_test, y_test))

