#!/usr/bin/env python
# coding: utf-8

# In[228]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")

print(data)

#Check the dataset
data.info()

#Removing the null values
data.dropna(inplace=True)

#Check the data without the null values
data.info()

from sklearn.model_selection import train_test_split
#Split the dataset in X & y
X = data.drop(['median_house_value'] , axis=1)

y = data['median_house_value']

print(X)
print(y)

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#Join two training dataset
train_data = X_train.join(y_train)

print(train_data)

train_data.hist(figsize=(15, 8))


# In[229]:


train_data.corr()


# In[230]:


plt.figure(figsize=(15,9))
sns.heatmap(train_data.corr(),annot=True)


# In[231]:


train_data['total_rooms'] = np.log(train_data['total_rooms'] +1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] +1)
train_data['population'] = np.log(train_data['population'] +1)
train_data['households'] = np.log(train_data['households'] +1)


# In[232]:


train_data.hist(figsize=(15, 9))


# In[233]:


train_data.ocean_proximity.value_counts()


# In[234]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)


# In[235]:


train_data


# In[236]:


plt.figure(figsize=(15,9))
sns.heatmap(train_data.corr(),annot=True)


# In[237]:


plt.figure(figsize=(15,9))
sns.scatterplot(x="latitude", y= "longitude", data=train_data, hue="median_house_value")


# In[238]:


train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']


# In[239]:


plt.figure(figsize=(15,9))
sns.heatmap(train_data.corr(),annot=True)


# In[240]:



from sklearn.linear_model import LinearRegression

X_train, y_train = train_data.drop(['median_house_value'], axis =1), train_data['median_house_value']

reg = LinearRegression()

reg.fit(X_train, y_train)


# In[243]:


test_data = X_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] +1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] +1)
test_data['population'] = np.log(test_data['population'] +1)
test_data['households'] = np.log(test_data['households'] +1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms'] 
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']


X_test, y_test = test_data.drop(['median_house_value'], axis =1), test_data['median_house_value']


# In[244]:


test_data


# In[245]:


reg.score(X_test, y_test)


# In[262]:


y_pred = reg.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()


# In[254]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train, y_train)


# In[257]:


RandomForest_prediction = forest.score(X_test, y_test)


# In[258]:


RandomForest_prediction


# In[261]:


y_pred = forest.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values")
plt.show()

