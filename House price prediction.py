#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install xgboost


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[4]:


house_price_dataset=pd.read_csv("Houseprice.csv")


# In[5]:


print(house_price_dataset)


# In[6]:


# Print First 5 rows of our DataFrame
house_price_dataset.head()


# In[7]:


# checking the number of rows and Columns in the data frame
house_price_dataset.shape


# In[8]:


house_price_dataset.isnull()


# In[9]:


# check for missing values
house_price_dataset.isnull().sum()


# In[10]:


# statistical measures of the dataset
house_price_dataset.describe()


# In[11]:


correlation=house_price_dataset.corr()


# In[12]:


# constructing a heatmap to nderstand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='coolwarm')


# In[15]:


X = house_price_dataset.drop(['MEDV'], axis=1)
Y = house_price_dataset['MEDV']


# In[16]:


print(X)
print(Y)


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)


# In[18]:


model = XGBRegressor()


# In[19]:


model.fit(X_train, Y_train)


# In[20]:


training_data_prediction = model.predict(X_train)

print(training_data_prediction)


# In[22]:


score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[23]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()


# In[24]:


test_data_prediction = model.predict(X_test)


# In[25]:


score_1 = metrics.r2_score(Y_test, test_data_prediction)

score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)