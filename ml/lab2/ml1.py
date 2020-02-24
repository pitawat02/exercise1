# Stock data
import quandl
import datetime
# Analyzing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


df = quandl.get('WIKI/GOOGL')
df = df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'])
df=(df-df.min())/(df.max()-df.min())
next30 = df['Close'].tolist()
for i in range(30):
    next30.pop(0)
    next30.append('0')
df['Next30'] = next30
for i in range(30):
    df = df.drop(df.index[len(df)-1])

for i in range(len(df)-60):
    df = df.drop(df.index[0])

x = df.drop(columns='Next30')
y = df.drop(columns='Close')
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# ตั้งค่าขนาดพื้นที่ภาพ
plt.figure(figsize=(15,7))
# scatter plot ความสัมพันธ์ของค่า X_train, y_train และ X_test, y_test
plt.scatter(X_train, y_train, marker='o', color='blue')
plt.scatter(X_test, y_test, marker='o', color='red')
plt.show()




kf = model_selection.KFold(n_splits=10,random_state=1, shuffle=True)
# Linear Regression Model
LRM = LinearRegression()
c_val = 1000
gmm = 0.1
svr_lin = SVR(kernel='linear', C=c_val)
svr_rbf = SVR(kernel='rbf', C=c_val, gamma=gmm)
svr_poly = SVR(kernel='poly', C=c_val, degree=2)


score0 = model_selection.cross_val_score(LRM,X_train,np.ravel(y_train),cv=kf)
score1 = model_selection.cross_val_score(svr_lin,X_train,np.ravel(y_train),cv=kf)
score2 = model_selection.cross_val_score(svr_rbf,X_train,np.ravel(y_train),cv=kf)
score3 = model_selection.cross_val_score(svr_poly,X_train,np.ravel(y_train),cv=kf)

print(score0.std())
print(score1)
print(score2)
print(score3)
'''
LRM = LRM.fit()
svr_lin = svr_lin.fit()
svr_rbf = svr_rbf.fit()
svr_poly= svr_poly.fit()
'''