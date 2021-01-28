#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn

seaborn.set()


# In[2]:


#Loading the data from the CSV file into a dataframe 
df1= pd.read_csv('dog_rates_tweets.csv')
df1


# In[3]:


data = df1.text.str.extract(r'(\d+(\.\d+)?)/10')
#Dropping all rows containing Nan
data.dropna(subset=[0], axis = 0 , inplace= True)
#Converting object datatype of the column to float datatype
data=pd.to_numeric(data[0])
data


# In[4]:


#Cleaning the data further
mask=data<25
data=data[mask]
data


# In[5]:


df1['Ratings']=data
#To display the data without Nan
df1 = df1.dropna()
df1
#df1.info()


# In[6]:


# Converting created_at column to a datetime value
df1['created_at']= pd.to_datetime(df1['created_at'],format="%Y-%m-%d %H:%M:%S") 
df1


# In[7]:


def to_timestamp(t):
    return t.timestamp()
df1['timestamp'] = df1['created_at'].apply(to_timestamp)
df1


# In[8]:


fit = stats.linregress(df1['timestamp'], df1['Ratings'])
fit.slope, fit.intercept


# In[9]:


plt.xticks(rotation=25)
x=df1['created_at'].values
y1= df1['Ratings']
plt.plot(x,y1, 'b.', alpha=0.5)
y2=df1['timestamp']*fit.slope + fit.intercept
plt.plot(x, y2, 'r-', linewidth=3)
plt.legend(['Rating','Best-fit line'],prop={'size':11})
plt.xlabel('Date')
plt.ylabel('Rating')
plt.show()


# In[10]:


pval=fit.pvalue
print(pval)


# In[11]:


#From lecture notes
y= df1['Ratings']
x= df1['timestamp']
residuals = y - (fit.slope*x + fit.intercept)


# In[12]:


plt.hist(residuals)
plt.xlabel('Residual')
plt.ylabel('No. of tweets')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




