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


# In[5]:


kf = model_selection.KFold(n_splits=10,random_state=42, shuffle=True)
# Linear Regression Model
LRM = LinearRegression()
c_val = 1000
gmm = 0.1
svr_lin = SVR(kernel='linear', C=c_val)
svr_rbf = SVR(kernel='rbf', C=c_val, gamma=gmm)
svr_poly = SVR(kernel='poly', C=c_val, degree=2)


score0 = model_selection.cross_val_score(LRM,X_train,y_train,cv=kf)
score1 = model_selection.cross_val_score(svr_lin,X_train,y_train,cv=kf)
score2 = model_selection.cross_val_score(svr_rbf,X_train,y_train,cv=kf)
score3 = model_selection.cross_val_score(svr_poly,X_train,y_train,cv=kf)


# In[9]:


close_validate = test60['Close'].to_numpy().reshape(60,1)
next30_validate = test60['Next30'].to_numpy()


# In[10]:


model0 = LRM.fit(X_train,y_train)
lrm_ans = model0.predict(X_test)
lrm_ans_test = model0.predict(close_validate)

model1 = svr_lin.fit(X_train,y_train)
svr_lin_ans = model1.predict(X_test)
svr_lin_ans_test = model1.predict(close_validate)

model2 = svr_rbf.fit(X_train,y_train)
svr_rbf_ans = model2.predict(X_test)
svr_rbf_ans_test = model2.predict(close_validate)

model3 = svr_poly.fit(X_train,y_train)
svr_poly_ans = model3.predict(X_test)
svr_poly_ans_test = model3.predict(close_validate)


# In[11]:



# In[12]:


c_param = [0.1,1,50,1000]
gamma = [0.1,0.5,1]
tune_parameters = [{'kernel':['rbf'],'C':c_param,'gamma':gamma}]

model = SVR()
cv_results = model_selection.GridSearchCV(model , tune_parameters ,cv=kf)
cv_results.fit(X_train,y_train)
print('BEST PARAMETER: ' , cv_results.best_params_, ' / ' , 'BEST SCORE: ',cv_results.best_score_)
#cv_results.to_csv('cv_results.csv')


# In[13]:


best = SVR(kernel='rbf',C=1,gamma=0.5)
best.fit(X_train,y_train)
best_result = best.predict(X_test)
best_result_test = best.predict(close_validate)

val_MSE= metrics.mean_squared_error(y_test,best_result)
val_rsquare = metrics.r2_score(y_test,best_result)
test_MSE = metrics.mean_squared_error(next30_validate,best_result_test)
test_rsquare =metrics.r2_score(next30_validate,best_result_test)

print('MSE validate = ' , val_MSE,
      '/ R-square validate = ' ,val_rsquare,
      '/ MSE test = ' ,test_MSE,
      '/ R-square test = ' ,test_rsquare)


# In[14]:


plt.figure(figsize=(10,7))
plt.scatter(X_test,lrm_ans,c='red',label = 'LRM')
plt.scatter(X_test,svr_lin_ans,c='yellow',label = 'SVR linear')
plt.scatter(X_test,svr_rbf_ans,c='green',label = 'SVR RBF')
plt.scatter(X_test,svr_poly_ans,c='orange',label = 'SVR POLY')
plt.scatter(X_test,best_result,c='black',label = 'best predict')
plt.scatter(X_test,y_test,c='blue',label = 'Actual')
plt.legend()
plt.title('Validate')

plt.show()

plt.figure(figsize=(10,7))
plt.scatter(close_validate, lrm_ans_test,c='red',label = 'LRM')
plt.scatter(close_validate ,svr_lin_ans_test,c='yellow',label = 'SVR linear')
plt.scatter(close_validate ,svr_rbf_ans_test,c='green',label = 'SVR RBF')
plt.scatter(close_validate ,svr_poly_ans_test,c='orange',label = 'SVR POLY')
plt.scatter(close_validate ,best_result_test,c='black',label = 'best predict')
plt.scatter(close_validate ,next30_validate,c='blue',label = 'Actual')
plt.legend()
plt.title('Test')

plt.show()

