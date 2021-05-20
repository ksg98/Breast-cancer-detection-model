#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
#breast cancer jupyter notebook file
#karamjeet Singh


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


#importing datset
from sklearn.datasets import load_breast_cancer


# In[8]:


cancer = load_breast_cancer()


# In[9]:


cancer


# In[10]:


cancer.keys()


# In[11]:


print(cancer['DESCR'])


# In[12]:


print(cancer['target'])


# In[13]:


print(cancer['feature_names'])


# In[14]:


cancer['data'].shape


# In[17]:


#structring the data into dataFrames to be underable
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns =np.append(cancer['feature_names'],['target']))


# In[18]:


df_cancer.head()


# In[19]:


df_cancer.tail()


# In[22]:


# Paramater plot , 5 taken
sns.pairplot(df_cancer, hue = 'target' ,vars = ['mean radius','mean texture','mean area','mean perimeter','mean smoothness'])


# In[23]:


sns.countplot(df_cancer['target'])


# In[24]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[26]:


plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)


# In[64]:


#Training the model
#dropping target colum
X = df_cancer.drop(['target'], axis =1)


# In[65]:


X


# In[29]:


y = df_cancer['target']


# In[30]:


y


# In[31]:


from sklearn.model_selection import train_test_split


# In[66]:


X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[67]:


X_train


# In[34]:


y_train


# In[68]:


X_test


# In[36]:


from sklearn.svm import SVC


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix


# In[46]:


svc_model = SVC()


# In[69]:


svc_model.fit(X_train, y_train)


# In[48]:


y_pred = svc_model.predict(x_test)


# In[49]:


y_pred


# In[53]:


cm = confusion_matrix(y_test, y_pred)


# In[54]:


sns.heatmap(cm , annot =True)


# In[55]:


#improving the model using unity based  data normalisation


# In[70]:


min_train = X_train.min()


# In[71]:


range_train = (X_train-min_train).max()


# In[73]:


X_train_scaled = (X_train-min_train)/range_train


# In[74]:


sns.scatterplot(x = X_train['mean area'],y = X_train['mean smoothness'], hue = y_train)


# In[75]:


sns.scatterplot(x = X_train_scaled['mean area'],y = X_train_scaled['mean smoothness'], hue = y_train)


# In[77]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[79]:


svc_model.fit(X_train_scaled, y_train)


# In[83]:


y_pred = svc_model.predict(X_test_scaled)


# In[85]:


cm = confusion_matrix(y_test, y_pred)


# In[86]:


sns.heatmap(cm, annot = True)


# In[87]:


#we left with 4
print(classification_report(y_test,y_pred))


# In[94]:


#we got 96% percent
#now we will search for best parameter, let sklean chhose our gamma parameters

param_grid ={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']} 


# In[96]:


from sklearn.model_selection import GridSearchCV


# In[97]:


grid = GridSearchCV(SVC(),param_grid,refit = True, verbose = 4)


# In[99]:


grid.fit(X_train_scaled,y_train)


# In[100]:


grid.best_params_


# In[101]:


# we got our best parameters

grid_pred = grid.predict(X_test_scaled)


# In[104]:


cm = confusion_matrix(y_test, grid_pred)


# In[105]:


sns.heatmap(cm, annot=True)


# In[106]:


# we left with AGAIN
print(classification_report(y_test,grid_pred))


# 
