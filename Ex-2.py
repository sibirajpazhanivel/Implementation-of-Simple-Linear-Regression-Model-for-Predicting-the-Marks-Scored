#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()


# In[4]:


df.tail()


# In[6]:


#segregating data to variables
X=df.iloc[:,:-1].values
X


# In[8]:


Y=df.iloc[:,1].values
Y


# In[10]:


#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# In[11]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)


# In[12]:


#displaying predicted values
Y_pred


# In[13]:


Y_test


# In[14]:


#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# In[15]:


plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# In[17]:


mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)


# In[18]:


mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)


# In[19]:


rmse=np.sqrt(mse)
print('RMSE = ',rmse)


# In[ ]:




