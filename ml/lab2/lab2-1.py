# In[1]:


# Stock data
import quandl
import datetime
# Analyzing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import preprocessing


# In[2]:


df = quandl.get('WIKI/GOOG')
df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'])
df=(df-df.min())/(df.max()-df.min())
next30 = df['Close'].tolist()
for i in range(30):
    next30.pop(0)
    next30.append('0')
df['Next30'] = next30
for i in range(30):
    df = df.drop(df.index[len(df)-1])

test60 = df.tail(60)
for i in range(60):
    df = df.drop(df.index[len(df)-1])
    


# In[18]:


x = df.drop(['Next30'],axis = 1).to_numpy()
y = df.drop(['Close'],axis = 1).to_numpy()
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)


# In[20]:


# ตั้งค่าขนาดพื้นที่ภาพ
plt.figure(figsize=(30,10))
# scatter plot ความสัมพันธ์ของค่า X_train, y_train และ X_test, y_test
plt.scatter(X_train, y_train, marker='o', color='blue', label = 'train')
plt.scatter(X_test, y_test, marker='o', color='red', label = 'test')
plt.title('X_train, y_train and X_test, y_test')
plt.legend()
plt.show()