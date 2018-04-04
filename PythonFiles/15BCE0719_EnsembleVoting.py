
# coding: utf-8

# ## Introduction
# Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms.
# 
# It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data

# In[1]:

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


# In[4]:

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
print "Length of dataset is -",dataframe.shape[0]
print "No of columns are - ",dataframe.shape[1]
print "\n\n",dataframe


# In[22]:

array = dataframe.values
print array
print array.shape
X = array[:,0:8]
# First part denotes how many rows we want. Second part denotes how many columns.
Y = array[:,8]


# In[24]:

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models


# ## Using 3 models
# - ### 1. Logistic Regression
# 

# In[25]:

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))


# - ### 2. Decision Tree

# In[26]:


model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))


# - ### 3. SVM

# In[27]:


model3 = SVC()
estimators.append(('svm', model3))


# In[34]:


# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print("Here is the score - ",results.mean()*100)

