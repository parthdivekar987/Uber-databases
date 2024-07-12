#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("uber.csv")
df


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


df =df.drop(['Unnamed: 0', 'key'], axis =1)


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(), inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(), inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.dtypes


# In[17]:


df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce')


# In[18]:


df.dtypes


# In[19]:


df= df.assign(hour = df.pickup_datetime.dt.hour,
              day = df.pickup_datetime.dt.day,
              month = df.pickup_datetime.dt.month,
              year = df.pickup_datetime.dt.year, 
              dayofweek = df.pickup_datetime.dt.dayofweek)


# In[20]:


df.head


# In[21]:


df = df.drop('pickup_datetime', axis =1)


# In[22]:


df.head


# In[23]:


df.dtypes


# In[25]:


df.plot(kind="box", subplots = True,layout =(7,2),figsize=(15,20))


# In[27]:


def remove_outlier(df1, col):
    Q1 =df1[col].quantile(0.25)
    Q3= df1[col].quantile (0.75)
    IQR= Q3 - Q1
    lower_whisker= Q1-1.5*IQR
    upper_whisker= Q3+1.5*IQR
    df[col] = np.clip(df1[col], lower_whisker, upper_whisker)
    return df1
def treat_outliers_all(df1, col_list):
    for c in col_list:
        df1 = remove_outlier(df, c)
    return df1


# In[28]:


df = treat_outliers_all(df,df.iloc[:, 0::])


# In[29]:


df.plot(kind = "box",subplots = True,layout=(7,2),figsize=(15,20))


# In[30]:


pip install haversine


# In[38]:


import haversine as hs
travel_dist = []
for pos in range(len(df['pickup_longitude'])):
    long1,lati1,long2,lati2 = [df['pickup_longitude'][pos],df['pickup_latitude'][pos],df['dropoff_longitude'][pos],df['dropoff_latitude'][pos]]
    loc1 =(lati1,long1)
    loc2 =(lati2,long2)
    c = hs.haversine(loc1,loc2)
    travel_dist.append(c)
                                                                                                                      
print(travel_dist)
df['dist_travel_km']= travel_dist
df.head


# In[39]:


df = df.loc[(df.dist_travel_km>=1) |(df.dist_travel_km<=130) ]
print("Remaining obervation:" , df.shape)


# In[47]:


incorrect_coordinates =df.loc[(df.pickup_latitude>90) | (df.pickup_latitude< -90)|
                              (df.dropoff_latitude>90) | (df.dropoff_latitude< -90) |
                              (df.pickup_longitude>180)| (df.pickup_longitude< -180)|
                              (df.dropoff_latitude>90) | (df.dropoff_latitude< -90) 
                             ]                            
                             


# In[48]:


df.drop(incorrect_coordinates, inplace = True, errors ='ignore')


# In[49]:


df.head()


# In[50]:


df.isnull().sum()


# In[51]:


sns.heatmap(df.isnull())


# In[52]:


corr = df.corr()


# In[53]:


corr


# In[55]:


fig,axis = plt.subplots(figsize= (10,6))
sns.heatmap(df.corr(), annot = True)


# In[56]:


df.dtypes


# In[58]:


x = df[['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude', 'passenger_count', 'hour', 'day','month', 'year','dayofweek','dist_travel_km']]


# In[59]:


y = df['fare_amount']


# In[60]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.33)


# In[61]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()


# In[62]:


regression.fit(x_train,y_train)


# In[64]:


regression.intercept_


# In[65]:


regression.coef_


# In[66]:


prediction = regression.predict(x_test)


# In[67]:


print(prediction)


# In[68]:


y_test


# In[69]:


from sklearn.metrics import r2_score


# In[70]:


r2_score(y_test,prediction)


# In[71]:


from sklearn.metrics import mean_squared_error


# In[72]:


MSE = mean_squared_error(y_test,prediction)


# In[73]:


MSE


# In[ ]:




