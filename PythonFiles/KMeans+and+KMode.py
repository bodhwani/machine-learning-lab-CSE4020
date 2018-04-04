
# coding: utf-8

# # K-Means
# ### 1. Using make_blobs dataset :

# In[61]:

import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing,cross_validation,neighbors
from sklearn.metrics import accuracy_score
get_ipython().magic(u'matplotlib inline')


# In[62]:


plt.rcParams['figure.figsize'] = (16, 9)

# Creating a sample dataset with 4 clusters
X, y = make_blobs(n_samples=800, n_features=3, centers=4)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])


# In[63]:


kmeans = KMeans(n_clusters=4)

kmeans = kmeans.fit(X)

labels = kmeans.predict(X)
C = kmeans.cluster_centers_
print C



# In[64]:

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)


# ### 2. Using CSV Dataset  :

# In[65]:

import pandas as pd
dataset = pd.read_csv('knndataset.csv')
dataset.head()

X = dataset.as_matrix(['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'])
Y = dataset.as_matrix(['Species'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y, test_size=0.2, random_state=100)


# In[66]:

k_means = KMeans(n_clusters=2)
k_means.fit(X_train)
print(k_means.labels_[:])
# print(y_train[:])


# In[67]:

print(k_means.predict(X_test))

print("Labels are ",k_means.labels_)

print(y_test[:])


# # K Mode

# In[68]:

import numpy as np
from kmodes.kmodes import KModes

# random categorical data
data = np.random.choice(20, (100, 10))

km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)

