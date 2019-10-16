#!/usr/bin/env python
# coding: utf-8

# In[120]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import missingno
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import random
from sklearn import model_selection, tree, preprocessing, metrics, linear_model 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


# In[68]:


train = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')


# In[69]:


train.head()


# In[70]:


test.head()


# In[71]:


train.describe()


# In[72]:


missingno.matrix(train, figsize = (30,10))


# In[73]:


test.describe()


# In[74]:


missingno.matrix(test, figsize = (30,10))


# In[75]:


def find_missing_no(df, columns):
    missing_val = {}
    print('Number of Missing at each column')
    length_df = len(df)
    for i in columns:
        total_val = df[i].value_counts().sum()
        missing_val[i] = length_df - total_val 
    print(missing_val)

find_missing_no(train, columns=train.columns)


# In[76]:


new_df = pd.DataFrame()


# In[77]:


train.isnull().sum()


# In[78]:


print(train['Year of Record'].isnull().sum())
print(test['Year of Record'].isnull().sum())


# In[79]:


train['Year of Record'].fillna(train['Year of Record'].mode()[0], inplace = True)
test['Year of Record'].fillna(test['Year of Record'].mode()[0], inplace =True)
print(train.head())


# In[80]:


print(train['Year of Record'].isnull().sum())
print(test['Year of Record'].isnull().sum())


# In[81]:


print(train['Gender'].value_counts())


# In[82]:


train['Gender'].fillna('unknown', inplace=True)
test['Gender'].fillna('unknown', inplace=True)


# In[83]:


train['Age'].fillna(train['Age'].mode()[0],inplace=True)
test['Age'].fillna(test['Age'].mode()[0], inplace=True)


# In[84]:


print(train['Profession'].isnull().sum())


# In[85]:


print(train['Profession'].value_counts())


# In[86]:


train['Profession'].fillna(train['Profession'].mode()[0], inplace=True)
test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
print(train['Profession'].value_counts())


# In[87]:


print(train['University Degree'].value_counts())


# In[88]:


train['University Degree'].fillna(train['University Degree'].mode()[0], inplace=True)
test['University Degree'].fillna(test['University Degree'].mode()[0], inplace=True)
print(train['University Degree'].value_counts())


# In[89]:


train['Hair Color'].fillna('Unknown', inplace=True)
test['Hair Color'].fillna('Unknown', inplace=True)


# In[90]:


print(train['Hair Color'].value_counts())


# In[91]:


find_missing_no(train, columns=train.columns)


# In[92]:


find_missing_no(test, columns=test.columns)


# In[93]:


train['flag'] = 1
test['flag'] = 0


# In[94]:


together_df = pd.DataFrame()


# 
# 
# **Creating a dataframe with combined test and train dataset
# Doing this so that the features in both test and train datasets are the same after one hot encoding**

# In[95]:


together_df = pd.concat([train,test])
le = preprocessing.LabelEncoder()


# In[96]:


find_missing_no(test, columns=test.columns)


# In[97]:


le.fit(together_df['Profession'])
together_df['Profession'] = le.transform(together_df['Profession'])

gender_onehot = pd.get_dummies(together_df['Gender'],prefix='gender')
country_onehot = pd.get_dummies(together_df['Country'],prefix='country')
deg_onehot = pd.get_dummies(together_df['University Degree'],prefix='deg')
haircol_onehot = pd.get_dummies(together_df['Hair Color'],prefix='haircol')


# In[98]:


together_df = pd.concat([together_df, gender_onehot, country_onehot, deg_onehot, haircol_onehot], axis=1)


# In[99]:


together_df.head()


# In[100]:


together_df.drop(['Country','Gender','Hair Color','Income','University Degree'],axis=1,inplace=True)


# In[101]:


together_df.head()


# In[102]:


train_df = together_df[together_df['flag'] == 1]
test_df = together_df[together_df['flag'] == 0]


# In[103]:


print(train_df.shape)
print(test_df.shape)


# In[104]:


train_df.head()


# In[105]:


train_df.drop(['flag'],axis=1,inplace=True)


# In[106]:


train_df.set_index('Instance')


# In[107]:


test_df.drop(['flag'],axis=1,inplace=True)


# In[108]:


test_df.set_index('Instance')


# In[109]:


test_df.drop(['Income in EUR'],axis=1,inplace=True)


# In[110]:


train_df.shape


# In[111]:


#Handling outliers
temp = train_df.sort_values(by=['Income in EUR'])
plt.scatter(temp['Age'], temp['Income in EUR'])
plt.show()


# In[112]:


temp = temp[:-8]
temp.shape


# In[113]:


plt.scatter(temp['Age'], temp['Income in EUR'])
plt.show()


# In[114]:


train.data = pd.DataFrame()
train_x = temp.drop('Income in EUR', axis=1)
train_y = temp['Income in EUR']


# In[115]:


regr_etr = ExtraTreesRegressor(n_estimators = 10, random_state = 0)
model_etr = regr_etr.fit(train_x, train_y)
acc = (model_etr.score(train_x, train_y))


# In[116]:




regr = LinearRegression()
model = regr.fit(train_x, train_y) 
acc = (model.score(train_x, train_y))


# In[117]:


test_dataset =regr_etr.predict(test_df)
filename = 'submission.csv'
pd.DataFrame({'Instance': test_df['Instance'], 'Income': test_dataset}).to_csv(filename, index=False)
print(acc)


# In[ ]:




