#!/usr/bin/env python
# coding: utf-8
#importing all the necessary libraries
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

#reading the data files
train = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

#testing if your object has the right type of data in it by using .head
train.head()
test.head()

#basic statistical details like percentile, mean, std etc.
train.describe()

#nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion.
missingno.matrix(train, figsize = (30,10))

test.describe()

missingno.matrix(test, figsize = (30,10))

#function to find missing values
def find_missing_no(df, columns):
    missing_val = {}
    print('Number of Missing at each column')
    length_df = len(df)
    for i in columns:
        total_val = df[i].value_counts().sum()
        missing_val[i] = length_df - total_val 
    print(missing_val)

find_missing_no(train, columns=train.columns)

#defined dataframe
new_df = pd.DataFrame()

#counts and displays the sum of null values
train.isnull().sum()

#YearofRecord
print(train['Year of Record'].isnull().sum())
print(test['Year of Record'].isnull().sum())

#Fill NA/NaN values using the specified method.
train['Year of Record'].fillna(train['Year of Record'].mode()[0], inplace = True)
test['Year of Record'].fillna(test['Year of Record'].mode()[0], inplace =True)
print(train.head())

print(train['Year of Record'].isnull().sum())
print(test['Year of Record'].isnull().sum())

#Gender
#Return a Series containing counts of unique values.
print(train['Gender'].value_counts())

train['Gender'].fillna('unknown', inplace=True)
test['Gender'].fillna('unknown', inplace=True)

#Age
train['Age'].fillna(train['Age'].mode()[0],inplace=True)
test['Age'].fillna(test['Age'].mode()[0], inplace=True)

#Profession
print(train['Profession'].isnull().sum())

print(train['Profession'].value_counts())

train['Profession'].fillna(train['Profession'].mode()[0], inplace=True)
test['Profession'].fillna(test['Profession'].mode()[0], inplace=True)
print(train['Profession'].value_counts())

#UniversityDegree
print(train['University Degree'].value_counts())

train['University Degree'].fillna(train['University Degree'].mode()[0], inplace=True)
test['University Degree'].fillna(test['University Degree'].mode()[0], inplace=True)
print(train['University Degree'].value_counts())

#HairColor
train['Hair Color'].fillna('Unknown', inplace=True)
test['Hair Color'].fillna('Unknown', inplace=True)

print(train['Hair Color'].value_counts())

#Calling Function to /find Missig Number
find_missing_no(train, columns=train.columns)
find_missing_no(test, columns=test.columns)

#to distinguish which data came from which data set
train['flag'] = 1
test['flag'] = 0

together_df = pd.DataFrame()

#Creating a combined dataframe of train and test data so that the features in both are the same
together_df = pd.concat([train,test])
le = preprocessing.LabelEncoder()

find_missing_no(test, columns=test.columns)

le.fit(together_df['Profession'])
together_df['Profession'] = le.transform(together_df['Profession'])

gender_onehot = pd.get_dummies(together_df['Gender'],prefix='gender')
country_onehot = pd.get_dummies(together_df['Country'],prefix='country')
deg_onehot = pd.get_dummies(together_df['University Degree'],prefix='deg')
haircol_onehot = pd.get_dummies(together_df['Hair Color'],prefix='haircol')

together_df = pd.concat([together_df, gender_onehot, country_onehot, deg_onehot, haircol_onehot], axis=1)

#Checking the new dataframe
together_df.head()

#Droping the Irelevant colums from the dataset
together_df.drop(['Country','Gender','Hair Color','Income','University Degree'],axis=1,inplace=True)

together_df.head()

#Distinguishing between train and test dataframes
train_df = together_df[together_df['flag'] == 1]
test_df = together_df[together_df['flag'] == 0]

#Return a tuple representing the dimensionality of the DataFrame.
print(train_df.shape)
print(test_df.shape)

train_df.head()
train_df.drop(['flag'],axis=1,inplace=True)

#Set the DataFrame index using existing column.
train_df.set_index('Instance')

test_df.drop(['flag'],axis=1,inplace=True)

test_df.set_index('Instance')

test_df.drop(['Income in EUR'],axis=1,inplace=True)

train_df.shape

#Handling outliers
temp = train_df.sort_values(by=['Income in EUR'])
plt.scatter(temp['Age'], temp['Income in EUR'])
plt.show()

temp = temp[:-8]
temp.shape

plt.scatter(temp['Age'], temp['Income in EUR'])
plt.show()


#defining train parametters for the regressor
train.data = pd.DataFrame()
train_x = temp.drop('Income in EUR', axis=1)
train_y = temp['Income in EUR']

#Defining the model for ExtraTree Regressor with n_estiators=80, which is bascially number of trees in the forest
regr_etr = ExtraTreesRegressor(n_estimators = 80, random_state = 0)
model_etr = regr_etr.fit(train_x, train_y)
acc = (model_etr.score(train_x, train_y))
regr = LinearRegression()
model = regr.fit(train_x, train_y) 
acc = (model.score(train_x, train_y))

#making a csv file for the submission
test_dataset =regr_etr.predict(test_df)
filename = 'submission.csv'
pd.DataFrame({'Instance': test_df['Instance'], 'Income': test_dataset}).to_csv(filename, index=False)
print(acc)
