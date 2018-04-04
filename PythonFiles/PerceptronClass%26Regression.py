
# coding: utf-8

# In[20]:

from sklearn import datasets
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
iris = datasets.load_iris()

# Assign petal length and petal width to X matrix (150 samples)
X = iris.data[:, [2, 3]]

# Class labels
y = iris.target


# In[21]:

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Optimization - Feature scaling
from sklearn.preprocessing import StandardScaler

# Initlalize a new StandardScaler object, sc
sc = StandardScaler()

# Using the fit method, estimate the sample mean and standard deviation for each feature demension. 
sc.fit(X_train)

# Transform both training and test sets using the sample mean and standard deviations
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[22]:


# Import the Perceptron implementation 
from sklearn.linear_model import Perceptron

# Initialize a new perceptron object, ppn.
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

# Train the model
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print('Misclassified samples: %d /' % (y_test != y_pred).sum(), y_test.size)
                          


# Different performance metrics
from sklearn.metrics import accuracy_score

# Calculate the classification accuracy of the perceptron on the test
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))



# In[19]:

from sklearn.neural_network import MLPRegressor
reg = MLPRegressor()
mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50,max_iter=150, shuffle=True, random_state=1,activation='relu')
mlp.fit(X_train_std, y_train)
y_pred2 = mlp.predict(X_test_std)
print y_pred2
from sklearn.metrics import accuracy_score

# Calculate the classification accuracy of the perceptron on the test
print('Accuracy: %.2f' % mlp.score(X_test_std, y_test))


# In[ ]:



