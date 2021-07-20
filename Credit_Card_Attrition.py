#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[53]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# # Taking Data and Data Analysis

# In[54]:


df1 = pd.read_csv("Banking_CreditCardAttrition_new.csv")
df1.head(10)


# In[55]:


df1.shape


# In[ ]:





# In[56]:


df1.isnull().sum()


# In[57]:


df1.info()


# In[58]:


df1.describe()


# In[59]:


df1['Total_Trans_Amt'] = df1['Trans_Amt_Oct12'] + df1['Trans_Amt_Nov12'] + df1['Trans_Amt_Dec12'] + df1['Trans_Amt_Jan13'] + df1['Trans_Amt_Feb13'] + df1['Trans_Amt_Mar13']
df1['Total_Trans_Count'] = df1['Trans_Count_Oct12'] + df1['Trans_Count_Nov12'] + df1['Trans_Count_Dec12'] + df1['Trans_Count_Jan13'] + df1['Trans_Count_Feb13'] + df1['Trans_Count_Mar13']


# In[ ]:





# In[60]:


df1.head()


# In[ ]:





# In[61]:


df1 = df1.drop(['Trans_Amt_Oct12','Trans_Amt_Nov12','Trans_Amt_Dec12','Trans_Amt_Jan13','Trans_Amt_Feb13','Trans_Amt_Mar13'
              ,'Trans_Count_Oct12','Trans_Count_Nov12','Trans_Count_Dec12','Trans_Count_Jan13',
              'Trans_Count_Feb13','Trans_Count_Mar13'], axis = 1)
df1.head()


# In[62]:


df1.info()


# In[63]:


df1.describe()


# In[64]:


df1.isnull().sum()


# In[65]:


df1['Customer_Age'].describe()


# In[66]:


df1['Credit_Limit'].describe()


# In[67]:


df1['Customer_Age'].fillna(df1['Customer_Age'].mean(),inplace = True)
df1['Credit_Limit'].fillna(df1['Credit_Limit'].mean(),inplace = True)


# In[68]:


df1.isnull().sum()


# In[69]:


sns.countplot(df1["Attrition_Flag"],color='blue').set(title = "Attrition Flag")
plt.show()


# In[70]:


fig = plt.figure(figsize=(12,5))
sns.countplot(x = 'Education_Level', data= df1,hue = 'Attrition_Flag')
plt.title('Distribution of Attrition based on education')
plt.show()


# In[101]:


sns.countplot(x = 'Gender', data= df1,hue='Attrition_Flag')
plt.title('Distribution of Attrition based on Gender')
plt.show()


# In[72]:


plt.figure(figsize=(10,5))
sns.countplot(x = 'Income_Category',data=df1, hue = 'Attrition_Flag')
plt.title('Distribution of Attrition based on Income')
plt.show()


# In[73]:


plt.figure(figsize=(10,5))
sns.countplot(x = 'Card_Category', data = df1,hue='Attrition_Flag')
plt.show()


# In[74]:


#df1.duplicated().sum()


# In[ ]:





# In[ ]:





# ## Checking Outliers by box plot

# In[75]:


def outliers_IQR(xx):
    quart_1, quart_3 = np.percentile(xx, [25, 75])
    IQR_value = quart_3 - quart_1
    lower_bound = quart_1 - (IQR_value * 1.5)
    upper_bound = quart_3 + (IQR_value * 1.5)
    sns.boxplot(x = xx.index, data=xx)


# In[ ]:





# In[76]:


df1.head()


# In[77]:


outliers_IQR(df1['Customer_Age'])
min_thresold, max_thresold = df1.Customer_Age.quantile([0.05,0.99])
min_thresold, max_thresold


# In[ ]:





# In[78]:


outliers_IQR(df1['Dependent_count'])
min_thresold, max_thresold = df1.Dependent_count.quantile([0.05,0.99])
min_thresold, max_thresold


# In[79]:


outliers_IQR(df1['Months_on_book'])
min_thresold, max_thresold = df1.Months_on_book.quantile([0.05,0.99])
min_thresold, max_thresold


# In[ ]:





# In[80]:


outliers_IQR(df1['Total_Relationship_Count'])
min_thresold, max_thresold = df1.Total_Relationship_Count.quantile([0.05,0.99])
min_thresold, max_thresold


# In[81]:


outliers_IQR(df1['Months_Inactive_12_mon'])
min_thresold, max_thresold = df1.Months_Inactive_12_mon.quantile([0.05,0.99])
min_thresold, max_thresold


# In[82]:


outliers_IQR(df1['Contacts_Count_12_mon'])
min_thresold, max_thresold = df1.Contacts_Count_12_mon.quantile([0.05,0.99])
min_thresold, max_thresold


# In[83]:


outliers_IQR(df1['Credit_Limit'])
min_thresold, max_thresold = df1.Credit_Limit.quantile([0.05,0.99])
min_thresold, max_thresold


# In[ ]:





# In[84]:


outliers_IQR(df1['Total_Revolving_Bal'])
min_thresold, max_thresold = df1.Total_Revolving_Bal.quantile([0.05,0.99])
min_thresold, max_thresold


# In[85]:


outliers_IQR(df1['Total_Trans_Amt'])
min_thresold, max_thresold = df1.Total_Trans_Amt.quantile([0.05,0.99])
min_thresold, max_thresold


# In[86]:


outliers_IQR(df1['Total_Trans_Count'])
min_thresold, max_thresold = df1.Total_Trans_Count.quantile([0.05,0.99])
min_thresold, max_thresold


# In[ ]:





# # Filling outliers

# In[87]:


def filling_outliers(yy):
    minm = yy.quantile(0.05)
    maxm = yy.quantile(0.99)
    yy.fillna(maxm)
    yy.fillna(minm)


# In[88]:


filling_outliers(df1.Months_on_book)
filling_outliers(df1.Months_Inactive_12_mon)
filling_outliers(df1.Contacts_Count_12_mon)
filling_outliers(df1.Credit_Limit)
filling_outliers(df1.Total_Trans_Amt)
filling_outliers(df1.Total_Trans_Count)


# In[89]:


df1.head(20)


# In[90]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def Calculate_VIF (data):
    VIF = pd.DataFrame()
    VIF['features'] = data.columns
    VIF['VIF_value'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return (VIF)


# In[91]:


New_data = df1[['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon',
                'Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Total_Trans_Amt','Total_Trans_Count']]
New_data.head()


# In[92]:


Calculate_VIF(New_data)


# In[ ]:





# In[93]:


New_data = New_data.drop(['Months_on_book','Customer_Age','Total_Trans_Count'], axis =1)
New_data.head()


# In[94]:


Calculate_VIF(New_data)


# In[95]:


df2 = df1[['Attrition_Flag','Dependent_count','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Total_Trans_Amt']]
df2.head()


# In[96]:


x = df2.drop(['Attrition_Flag'], axis =1)
y = df2['Attrition_Flag']


# # Logistic Regression

# In[97]:


from  sklearn.linear_model import LogisticRegression


# In[98]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[99]:



model_LR = LogisticRegression(max_iter=100, random_state=0)
model_LR.fit(X_train, Y_train)


# In[100]:


prediction = model_LR.predict(X_test)
print(classification_report(Y_test,prediction))

