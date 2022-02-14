#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
sns.set(rc={'figure.figsize':(11.7,8.27)})
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Final Data Sheet.csv")


# In[3]:


df.head()


# In[4]:


# These steps are required to remove the column names after every block (as they are not required)
# Start with the first column that is replaced by zero to get the exact index position for every row with column name
df["ACE_Bx"]=df["ACE_Bx"].replace("ACE_Bx",0)


# In[5]:


# and in the code below every other row with column name is replaced by zero (to identify )
df_emp=df[df["ACE_Bx"] == 0].replace(df.columns[1:],0)
df_emp


# In[6]:


# Remove the rows with zero ( the column names)
df=df.drop(index=df_emp.index.tolist()).reset_index(drop=True)


# In[7]:


# Convert the data types to float
print(df.dtypes)


# In[8]:


for x in df.columns.tolist():
    df[f"{x}"]=df[f"{x}"].astype(float)


# In[9]:


print(df.dtypes)


# In[10]:


df.insert(11, "Block", "Block1")
# add the block column just for identifcation 
# the block can also be used to avergae if required


# In[11]:


# LOGIC TO FILL IN THE DELAY_VALUE FOR EVERY BLOCK

initial_index=0
block_index=1
final_index=80
while final_index <= df.shape[0]:
    delay_value=df.iloc[initial_index,-1]
    df.iloc[initial_index:final_index,-1]=delay_value
    df.iloc[initial_index:final_index,-2]=f"Block {block_index}"
    initial_index+=80
    final_index+=80
    block_index+=1
    
    


# In[12]:


df.tail()


# In[ ]:





# In[13]:


# Now the values are filled and the preprocessing is done
# We can proceed with adding the heatmap and traning the models
df.isna().sum()


# In[14]:


# pearson correlation with the pandas corr()
features = df.iloc[:,0:-2]
sns.heatmap(features.corr(method="pearson"),annot=True,linewidth=0.5)


# In[15]:


# function to compute Average Delay
def computeAverageDelay(array):
    avg=[]
    initial_index=0
    final_index=80
    while final_index <= array.shape[0]:
        avg.append(array[initial_index:final_index].mean())
        initial_index+=80
        final_index+=80
    return (np.array(avg))
# evaluate
def evaluate(X_train,y_train,y_train_avg,y_pred_avg,model):
    accuracy=abs(model.score(X_train,y_train))
    rmse = abs(metrics.mean_squared_error(y_train_avg,y_pred_avg,squared=False))
    return (accuracy,rmse)


# In[16]:


# get and X and y
X=df.iloc[:,:-2]
y=df.iloc[:,-1]


# In[17]:


#Shuffle needs to be false here to make sure that it is divided block wise
# DIVIDE DATA INTO TRAIN AND TEST SPLIT
# OF 20 TRAIN AND 80 TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.20, shuffle=False)


# In[18]:


LR20=LinearRegression()
LR20.fit(X_train,y_train)


# In[19]:


y_pred_train = LR20.predict(X_train)
y_pred_train_avg= computeAverageDelay(y_pred_train)

y_pred_test = LR20.predict(X_test)
y_pred_test_avg= computeAverageDelay(y_pred_test)


# In[20]:


y_train_avg =computeAverageDelay(y_train)
y_test_avg =computeAverageDelay(y_test)
lr20_accuracy_train, lr20_rmse_train= evaluate(X_train,y_train,
                              y_train_avg,y_pred_train_avg,LR20)
lr20_accuracy_test, lr20_rmse_test= evaluate(X_test,y_test,
                              y_test_avg,y_pred_test_avg,LR20)


# In[21]:


# Next do the same process for 50 percent train size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.50, shuffle=False)

LR50=LinearRegression()
LR50.fit(X_train,y_train)

y_pred_train = LR50.predict(X_train)
y_pred_train_avg= computeAverageDelay(y_pred_train)

y_pred_test = LR50.predict(X_test)
y_pred_test_avg= computeAverageDelay(y_pred_test)

y_train_avg =computeAverageDelay(y_train)
y_test_avg =computeAverageDelay(y_test)
lr50_accuracy_train, lr50_rmse_train= evaluate(X_train,y_train,
                              y_train_avg,y_pred_train_avg,LR50)
lr50_accuracy_test, lr50_rmse_test= evaluate(X_test,y_test,
                              y_test_avg,y_pred_test_avg,LR50)


# In[22]:


# same process for the 70 percent data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, shuffle=False)

LR70=LinearRegression()
LR70.fit(X_train,y_train)

y_pred_train = LR70.predict(X_train)
y_pred_train_avg= computeAverageDelay(y_pred_train)

y_pred_test = LR70.predict(X_test)
y_pred_test_avg= computeAverageDelay(y_pred_test)

y_train_avg =computeAverageDelay(y_train)
y_test_avg =computeAverageDelay(y_test)
lr70_accuracy_train, lr70_rmse_train= evaluate(X_train,y_train,
                              y_train_avg,y_pred_train_avg,LR70)
lr70_accuracy_test, lr70_rmse_test= evaluate(X_test,y_test,
                              y_test_avg,y_pred_test_avg,LR70)


# In[23]:


# plot the accuracy and rmse for the models
# first create the dataframe for it
trainSet_metrics = pd.DataFrame({"Train_Size":["20","50","70"],
             "Accuracy":[lr20_accuracy_train,lr50_accuracy_train,lr70_accuracy_train],
              "RMSE":[lr20_rmse_train,lr50_rmse_train,lr70_rmse_train]})
# use log scale to plot on graph
testSet_metrics = pd.DataFrame({"Train_Size":["20","50","70"],
             "Accuracy":np.log([lr20_accuracy_test,lr50_accuracy_test,lr70_accuracy_test]),
              "RMSE":np.log([lr20_rmse_test,lr50_rmse_test,lr70_rmse_test])})


# In[24]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11.7,8.27))
sns.barplot(x="Train_Size",y="Accuracy",data=trainSet_metrics,ax=ax1).set_title("Accuracy on train set")
sns.barplot(x="Train_Size",y="RMSE",data=trainSet_metrics,ax=ax2).set_title("RMSE on train set")


# In[25]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11.7,8.27))
sns.barplot(x="Train_Size",y="Accuracy",data=testSet_metrics,ax=ax1).set_title("Accuracy on test set")
sns.barplot(x="Train_Size",y="RMSE",data=testSet_metrics,ax=ax2).set_title("RMSE on test set")


# In[ ]:





# In[ ]:




