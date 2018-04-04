
# coding: utf-8

# In[1]:

from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
import pandas
import numpy as np


# In[4]:

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
print "Length of dataset is -",dataframe.shape[0]
print "No of columns are - ",dataframe.shape[1]
print "\n\n",dataframe


# In[5]:

array = dataframe.values
print array
print array.shape
X = array[:,0:8]
# First part denotes how many rows we want. Second part denotes how many columns.
Y = array[:,8]



# In[7]:

clf = OutputCodeClassifier(LinearSVC(random_state=0),
                           code_size=2, random_state=0)
clf.fit(X, Y)
pred =  clf.predict(X)
print "Here is the score - ",clf.score(X,Y)*100



