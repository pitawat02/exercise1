import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.api.types import is_numeric_dtype
import seaborn as sns; sns.set()

def remove_outlier(df):
	low = .01
	high = .99
	quantile_df = df.quantile([low, high])
	for i in list(df.columns):
	    if is_numeric_dtype(df[i]):
	       df = df[(df[i] > quantile_df.loc[low, i]) & (df[i] < quantile_df.loc[high, i])]
	return df

def zeromean(df):
    for i in list(df.columns):
        df[i] = df[i]-df[i].mean()
    return df

rawdata = pd.read_csv('watch_test2_sample.csv', parse_dates =["uts"], index_col ="uts")
df = pd.DataFrame(rawdata)
df = df.drop_duplicates()
df = df.dropna()
df = df.replace(0,np.nan)
df = df.replace(np.nan,df.mean())
df = df.drop(columns=['gps.y', 'gps.x'])
df = remove_outlier(df)

df = df.resample('20S', label='right', closed='right').mean()
df = df.interpolate(method='time' ,limit_direction = 'forward') 

df = df.drop(df.index[0:370])
df = df.drop(df.index[245:])

df = df.drop(columns=['compass','gyro.x','gyro.y','gyro.z','heartrate','light','pressure'])

df = zeromean(df)
cov = (df.T.dot(df))/(df.shape[0]-1)
eig_vals , eig_vec = np.linalg.eig(cov) 

idx = eig_vals.argsort()[::-1]   
eig_vals = eig_vals[idx]
eig_vec = eig_vec[:,idx]

s = pd.DataFrame.from_dict(eig_vals)
s.plot.bar()
plt.show()

ev1 = eig_vec[:,0]*np.sqrt(eig_vals[0])
ev2 = eig_vec[:,1]*np.sqrt(eig_vals[1])
ev3 = eig_vec[:,2]*np.sqrt(eig_vals[2])
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(141, projection='3d')
ax.plot(df['accelerateX'], df['accelerateY'], df['accelerateZ'], 'o', markersize=10, color='green', alpha=0.2)
ax.plot([df['accelerateX'].mean()], [df['accelerateY'].mean()], [df['accelerateZ'].mean()], 'o', markersize=10, color='red', alpha=0.5)
ax.plot([0, ev1[0]], [0, ev1[1]], [0, ev1[2]], color='red', alpha=0.8, lw=2)
ax.plot([0, ev2[0]], [0, ev2[1]], [0, ev2[2]], color='violet', alpha=0.8, lw=2)
ax.plot([0, ev3[0]], [0, ev3[1]], [0, ev3[2]], color='cyan', alpha=0.8, lw=2)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
ax.view_init(10,60)
plt.show()

df2 = df.dot(eig_vec)
df2.drop(columns=2)

columns = [0,1]
arr = df2[columns].to_numpy()
sns.heatmap(arr, cmap ='YlGn')
plt.show()
