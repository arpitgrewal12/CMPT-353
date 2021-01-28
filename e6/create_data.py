#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import time
from implementations import all_implementations


# In[2]:


#Initializing an empty data frame to store values later on
dataframe= pd.DataFrame()


# In[12]:



for sort in all_implementations:
    array = np.empty(75, dtype=float)
    #print(random_array)
    for i in range(75):
        lowest_integer=10000
        highest_integer=11001
        number_of_random_integers=1001
        random_array = np.random.randint(lowest_integer, highest_integer,number_of_random_integers)
        st = time.time()
        res = sort(random_array)
        en = time.time()
        total=en-st
        #print(total)
        array[i]=total
        #print(values[i])
        name_of_sort = sort.__name__
    dataframe[name_of_sort] = array
  

dataframe.to_csv('data.csv', index=False)


# In[15]:


print(dataframe)


