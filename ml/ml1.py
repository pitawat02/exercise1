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

rawdata = pd.read_csv('watch_test2_sample.csv', parse_dates =["uts"], index_col ="uts")
df = pd.DataFrame(rawdata)
df = df.drop_duplicates()
df = df.dropna()

df = df.replace(0,np.nan)
df = df.replace(np.nan,df.mean())
#map-------------------
BBox = [13.5480,13.6216,100.2775,100.3691]
map_im = plt.imread("map.png")
fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(df['gps.x'],df['gps.y'],zorder = 1,alpha = 0.5,c='r',s=20)
ax.set_title('Plotting Spantial Data on Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(map_im,zorder=0,extent = BBox,aspect='auto')
plt.show()
#---------------------
df = df.drop(columns=['gps.y', 'gps.x'])
df = remove_outlier(df)

df = df.resample('20S', label='right', closed='right').mean()
df = df.interpolate(method='time', limit_direction = 'forward') 

df = df.drop(df.index[0:370])
df = df.drop(df.index[245:])

df=(df-df.min())/(df.max()-df.min())

fig1 = plt.figure(figsize=(6,4))
ax = fig1.add_subplot(1,1,1, projection='3d')
ax.scatter(df[df.columns[0]],df[df.columns[1]],df[df.columns[2]],c='cyan',s=20,edgecolor='k')
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
ax.set_zlabel(df.columns[2])

fig2 = plt.figure(figsize=(6,4))
ay = fig2.add_subplot(1,1,1, projection='3d')
ay.scatter(df[df.columns[4]],df[df.columns[5]],df[df.columns[6]],c='violet',s=20,edgecolor='k')
ay.set_xlabel(df.columns[4])
ay.set_ylabel(df.columns[5])
ay.set_zlabel(df.columns[6])
plt.show()

#step5 stride4
stridedf = pd.concat(
    [  
        df.iloc[z:z+5] 
    for z in range(0,df.shape[0]-1,4)
    ]   
)

columns = ['accelerateX','accelerateY','accelerateZ','compass','heartrate']
plt.figure(1)
arr = df[columns].to_numpy()
sns.heatmap(arr,annot = False)
plt.figure(2)
arr2 = stridedf[columns].to_numpy()
sns.heatmap(arr2,annot = False, cmap = 'Blues')
plt.show()