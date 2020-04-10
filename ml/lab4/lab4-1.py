#import lib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from networkx import nx
from matplotlib.lines import Line2D
import seaborn as sns
import os, glob
get_ipython().run_line_magic('matplotlib', 'inline')


# In[113]:


#raw data
path = "ml-latest-small"
all_files = glob.glob(os.path.join(path,"*.csv"))
display(all_files)
for f in all_files:
    display(pd.read_csv(f,sep=','))


# In[114]:


#cleaning data
df_movies=pd.read_csv("ml-latest-small/movies.csv")
df_ratings=pd.read_csv("ml-latest-small/ratings.csv")
df_tags=pd.read_csv("ml-latest-small/tags.csv")


df_movies  = df_movies.dropna()
df_movies  = df_movies.drop_duplicates()

df_ratings  = df_ratings.dropna()
df_ratings  = df_ratings.drop_duplicates()

df_tags  = df_tags.dropna()
df_tags  = df_tags.drop_duplicates()


# In[115]:


ratings_yr = df_ratings
time = pd.to_datetime(df_ratings.timestamp, unit='s').dt.year
ratings_yr['year'] = time
ratings_yr


# In[116]:


count = 0
genres_list = []
for i in df_movies['genres']:
  x = i.split('|')
  for j in x:
    if j not in genres_list:
      genres_list.append(j)
    df_movies.at[count,j] = 1
  count+=1
  x = []


# In[117]:



split_data = df_movies["genres"].str.split("|")
data = split_data.to_list()
mlb = MultiLabelBinarizer()
new_df = pd.DataFrame(mlb.fit_transform(data),columns=mlb.classes_)
gen_df = new_df.copy()
new_df['movieId'] = df_movies['movieId'].to_numpy()
new_df['title'] = df_movies['title'].to_numpy()

cols = new_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols = cols[-1:] + cols[:-1]
new_df = new_df[cols]
new_df


# In[118]:


def ChangeLtS(list1):  
    string = ""  
    for em in list1:  
        string += em   
    return string  

cnt = 0
cnt_str = 0
string_year = []
listyear = []
for i in new_df['title']:
  for j in i:
    if j >='0' and j <='9' or j == '(' or j == ')':
      cnt_str += 1
      string_year.append(j)
    else:
      cnt_str = 0
      string_year = []
    if cnt_str == 6:
      if ChangeLtS(string_year)[1:5] not in listyear:
        listyear.append(ChangeLtS(string_year)[1:5])
      new_df.at[cnt,'year'] = ChangeLtS(string_year)[1:5]
  cnt += 1

new_df.dropna(inplace = True)
new_df['year'] = new_df['year'].astype(int)
sumyear = new_df.groupby('year').sum()
sumyear.reset_index(inplace = True)
sumyear.drop(['movieId'], axis = 1, inplace = True)

sumrate = df_ratings.drop(['userId', 'movieId', 'rating', 'timestamp'], axis = 1)
sumrate['temp'] = 1
sumrate = sumrate.groupby('year').sum()
sumrate.reset_index(inplace = True)


# In[119]:


plt.figure(figsize = (20,10))
plt.subplot(2, 2, 1)
legend = []
for i in listyear:
  legend.append(i)
  tempo = 0
  for i in genres_list:
    tempo += sumyear[i]
  plt.bar(x = sumyear['year'], height = tempo , width = 0.25)
plt.title('Released Movies/Year')
plt.xlabel('Year')
plt.ylabel('Released Movies')

plt.subplot(2, 2, 2)
plt.bar(x = sumrate['year'], height = sumrate['temp'], width = 0.25)
plt.title('Sum Ratings/Year')
plt.xlabel('Year')
plt.ylabel('Sum Ratings')
plt.show()


# In[120]:


#กราฟที่ 2: แสดงกราฟค่า จำนวนการให้rating ในแต่ละปี
ratings_yr.groupby(['year']).count().rating.plot.bar()


# In[121]:


x = []
y = []
for i in gen_df:
  temp = gen_df[i].value_counts()
  x.append(i)
  y.append(temp[1])


# In[122]:


#กราฟที่ 3: แสดงกราฟค่า จำนวน movies ในแต่ละ genre
plt.figure(figsize=(20,10))
sns.barplot(x, y, alpha=0.8)
plt.title('Movie Genres')
plt.ylabel('Number of Occurrences')
plt.xlabel('Genres')
plt.show()


# In[123]:


df_movies_yrs = new_df.groupby('year').sum()
df_movies_yrs.drop(['movieId'],axis = 1, inplace = True)
df_movies_yrs.reset_index(inplace = True)


# In[124]:


#- กราฟที่ 4: แสดงกราฟ (y-axis: stacked graph) ค่าจำนวน movie แต่ละ genre ในแต่ละปี (x-axis)
legend = []
plt.figure(figsize = (25,15))
for i in genres_list:
  legend.append(i)
  plt.bar(x = df_movies_yrs['year'], height = df_movies_yrs[i], width = 0.5)
plt.legend(legend)
plt.show()


# In[125]:


#กราฟที่ 5: แสดงกราฟ Histogram ของการกระจายของค่าเฉลี่ย movie rating ใน dataset

movie_ratings = df_ratings.groupby('movieId').rating.mean()
movie_ratings.plot.hist(bins=20).set_xlabel("Rating")