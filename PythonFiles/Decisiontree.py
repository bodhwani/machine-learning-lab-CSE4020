
# coding: utf-8

# In[104]:

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
df = pd.read_csv('dataset.csv')


# In[105]:

df.head()


# # How to handle missing values
# - use dropna as df.dropna(how="any" or "all", inplace=True)
# - use df.fillna(value = -99999,inplace=True)
# - or use df.fillna(method='ffill',inplace=True). So this takes previous values and fill them in in next nan value
# - or use df.fillna(method="bfill',inplace=True). So this takes future values and fill them in in next nan value

# In[106]:

df=df.dropna(axis=0, how='any')
len(df)


# In[107]:

df['Type'] = pd.Categorical(df.Type).codes
df['Reliability'] = pd.Categorical(df.Reliability).codes
df['Country'] = pd.Categorical(df.Country).codes


# **We can also use LabelEncoding or One Hot Encoding for converting categorical values into integers as shown below**

# In[98]:

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()
# le.fit(df['Type'])
# typeConvertor = le.transform(df['Type'])

# le.fit(df['Country'])
# countryConvertor = le.transform(df['Country'])

# le.fit(df['Reliability'])
# reliabilityConvertor = le.transform(df['Reliability'])
# print reliabilityConvertor
# print countryConvertor


# In[108]:

Y = df['Mileage']
X = df.drop(['Mileage','Name'],axis=1)


# In[113]:

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.15, random_state = 100)


# In[114]:

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)


# In[116]:

pred = clf.predict(X_test)
accuracy = accuracy_score(pred,y_test)
print("Accuracy is ",accuracy*100)


# In[ ]:



