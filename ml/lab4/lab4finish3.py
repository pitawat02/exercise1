#!/usr/bin/env python
# coding: utf-8

# In[112]:


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


# In[126]:


#2


# In[127]:


#2.1 สร้างข้อมูลความชอบของผู้ใช้แต่ละคน (user_ matrix)
Mcolumn = ['userId']
MId = int(df_ratings.iloc[:,1].max())
for i in range(1,MId+1):
    Mcolumn.append(i)

rate_mix =[]

for j in range(1,df_ratings.userId.max()+1):
    rowL = [0.0]*(MId+1)
    rowL[0] = j
    rate_mix.append(rowL)

for row in range(0,df_ratings.shape[0]):
    userId = int(df_ratings.iloc[row].userId)
    movieId = int(df_ratings.iloc[row].movieId)
    rating = float(df_ratings.iloc[row].rating)
    rate_mix[userId-1][movieId] = rating
    
user_matrix = pd.DataFrame(rate_mix,columns=Mcolumn).set_index(['userId'])
user_matrix


# In[128]:


#2.2 คำนวณความคล้ายกันของความชอบดูหนังของคู่ ‘userId’ ใดๆ


# In[129]:


#2.2.1 สุ่มหยิบข้อมูล user_matrix มาจำนวน nUser ไม่น้อยกว่า 20 คน
import random
nUser= 25
randomlist = []
for i in range(0,nUser):
    n = random.randint(1,user_matrix.shape[0]+1)
    randomlist.append(n)

randomlist.sort()

print(randomlist)
Mrandom_user = user_matrix.loc[randomlist]
Mrandom_user


# In[130]:


#2.2.2 คำนวณความคล้ายของความชอบ movie ของคู่ user โดยใช้ตัววัด cosine_similarity()
import sklearn.metrics
cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(Mrandom_user)
cosine_similarity = pd.DataFrame(cosine_similarity,columns = randomlist )
cosine_similarity['userId'] = randomlist
cosine_similarity = cosine_similarity.set_index('userId')
cosine_similarity


# In[131]:


#2.2.3 คำนวณความคล้ายของความชอบ movie ของคู่ user โดยใช้ตัววัด Pearson’s similarity()
pearson_similarity = Mrandom_user.T.corr ( method ='pearson' )
pearson_similarity


# In[132]:


#2.3 แสดงตารางรายการดังนี้


# In[133]:


#2.3.1 ตาราง user ที่มีความชอบคล้ายกันที่สุด 5 อันดับ

def insertionSort(arr): 
  
    # Traverse through 1 to len(arr) 
    for i in range(0, 5): 
  
        key1 = arr[i][0] 
        key2 = arr[i][1] 
        key3 = arr[i][2] 
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        j = i-1
        while j >=0 and key1 > arr[j][0] : 
                arr[j+1][0] = arr[j][0] 
                arr[j+1][1] = arr[j][1]
                arr[j+1][2] = arr[j][2]
                j -= 1
        arr[j+1][0] = key1 
        arr[j+1][1] = key2
        arr[j+1][2] = key3 


        
needed_highest = 5 # This is where your 100 would go
result = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
ccc = 0
for y in range(0, cosine_similarity.shape[0]):
    for x in range(y, cosine_similarity.shape[1]):
        num = cosine_similarity.iloc[y,x]
        if ccc <= 4 and num < 0.99999:
            result[4][0] = num
            ccc = ccc+1
            result[4][1] = randomlist[x]
            result[4][2] = randomlist[y]
            insertionSort(result)
        elif num < 0.99999:
            if result[4][0] < num : 
                result[4][0] = num
                result[4][1] = randomlist[x]
                result[4][2] = randomlist[y]
                insertionSort(result)

order_cosine_similarity = pd.DataFrame(result,columns =['cosine_similarity_score','userId1','userId2']).set_index(['userId1','userId2']) 
order_cosine_similarity


# In[134]:


result = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
ccc = 0
for y in range(0, pearson_similarity.shape[0]):
    for x in range(y, pearson_similarity.shape[1]):
        num = pearson_similarity.iloc[y,x]
        if ccc <= 4 and num < 0.99999:
            result[4][0] = num
            ccc = ccc+1
            result[4][1] = randomlist[x]
            result[4][2] = randomlist[y]
            insertionSort(result)
        elif num < 0.99999:
            if result[4][0] < num : 
                result[4][0] = num
                result[4][1] = randomlist[x]
                result[4][2] = randomlist[y]
                insertionSort(result)

order_pearson_similarity = pd.DataFrame(result,columns =['cosine_similarity_score','userId1','userId2']).set_index(['userId1','userId2']) 
order_pearson_similarity


# In[135]:


#2.3.2 ตาราง user ที่มีความชอบตรงกันข้ามกันที่สุด 5 อันดับ 

def insertionSort2(arr): 
  
    # Traverse through 1 to len(arr) 
    for i in range(0, 5): 
  
        key1 = arr[i][0] 
        key2 = arr[i][1] 
        key3 = arr[i][2] 
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        j = i-1
        while j >=0 and key1 < arr[j][0] : 
                arr[j+1][0] = arr[j][0] 
                arr[j+1][1] = arr[j][1]
                arr[j+1][2] = arr[j][2]
                j -= 1
        arr[j+1][0] = key1 
        arr[j+1][1] = key2
        arr[j+1][2] = key3 


needed_lowest = 5 # This is where your 100 would go
result = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
ccc = 0
for y in range(0, pearson_similarity.shape[0]):
    for x in range(y, pearson_similarity.shape[1]):
        num = pearson_similarity.iloc[y,x]
        if ccc <= 4 and num < 0.99999:
            result[4][0] = num
            ccc = ccc+1
            result[4][1] = randomlist[x]
            result[4][2] = randomlist[y]
            insertionSort2(result)
        elif num < 0.99999:
            if result[4][0] > num : 
                result[4][0] = num
                result[4][1] = randomlist[x]
                result[4][2] = randomlist[y]
                insertionSort2(result)

order_pearson_similarity = pd.DataFrame(result,columns =['pearson_similarity_score','userId1','userId2']).set_index(['userId1','userId2'])  
order_pearson_similarity


# In[136]:


#pearon sim
similar_pearson= []
top = 5

for row in range(0,pearson_similarity.shape[0]):
    for column in range(row,pearson_similarity.shape[1]):
        cell = pearson_similarity.iloc[row,column]
        spc = [0]*3
        spc[0] = randomlist[row]
        spc[1] = randomlist[column]
        spc[2] = cell
        if spc[0] != spc[1] :
            similar_pearson.append(spc)
        
similar_pearson_columns = ['userId1','userId2','pearson_similarity_score']

osp = pd.DataFrame(similar_pearson,columns = similar_pearson_columns).set_index(['userId1','userId2'])
osp_sort = osp.sort_values(by=['pearson_similarity_score'], ascending=False)
osp_sort.head(top)
osp_sort.tail(top)


# In[137]:


#2.3.3
heatmap_pearson = pd.pivot_table(osp, values='pearson_similarity_score', index=['userId1'], columns='userId2')

fig, ax = plt.subplots(figsize=(30,30)) 
sns.heatmap(heatmap_pearson,annot=True , ax=ax)


# In[138]:


#2.4 แสดงรูปภาพ


# In[139]:


#2.4 กราฟความคล้ายของความชอบ movie
from networkx import nx
from matplotlib.lines import Line2D

# Create New Graph
G = nx.Graph()

# Create #node = #user ใน Pearson’s similarity และใส่ label เป็น user_id ในแต่ละ node
for x in range(0,len(randomlist)):
    G.add_node(randomlist[x])

## Create #edge of graph ตามค่าใน Pearson’s similarity ที่เป็นตามเงื่อนไข 2 เงื่อนไขที่ตั้งไว้ข้างต้น คือแสดงสีเฉพาะ user ที่ชอบคล้ายกันเกิน Th 3 ระดับสีและชอบตรงข้ามกันเกิน Th 3 ระดับสี  

rank1 = 0.55
rank2 = 0.35
rank3 = 0.25

rank4 = 0.025
rank5 = 0.015
rank6 = 0.005

for x in range(0,len(randomlist)-1):
    for y in range(x, len(randomlist)-1):
        sav = osp.index.levels[0][x],osp.index.levels[1][y]
        v = osp.loc[sav].values
        if v > rank1 :
            G.add_edge( osp.index.levels[0][x],osp.index.levels[1][y]  , weight = 4 ,color='deepskyblue')
        elif v > rank2 :
            G.add_edge( osp.index.levels[0][x],osp.index.levels[1][y] , weight = 4,color='navy')
        elif v > rank3 :
            G.add_edge( osp.index.levels[0][x],osp.index.levels[1][y] , weight = 4 ,color='mediumpurple')
        elif v< rank6:
            G.add_edge( osp.index.levels[0][x],osp.index.levels[1][y] , weight = 2 ,color='brown')
        elif v < rank5 :
            G.add_edge( osp.index.levels[0][x],osp.index.levels[1][y], weight = 2 ,color='coral')
        elif v < rank4 :
            G.add_edge( osp.index.levels[0][x],osp.index.levels[1][y] , weight = 2 ,color='salmon')
        

edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
width = [G[u][v]['weight'] for u,v in edges]

nx.draw_circular(G,with_labels=True,edge_color=colors,node_size=500,font_size=12,font_color='white',width=width)


# In[140]:


#2.4.2 แนะนำของคนที่มีความชอบคล้ายกันที่สุด
suggestL =[]

for x in range(0,len(randomlist)):
    Vmaxi = 0
    Imaxi = 0
    for y in range(0,len(randomlist)):
        v = pearson_similarity.iloc[x,y]
        if v > Vmaxi and x !=y :
            Vmaxi = v
            Imaxi = y

    user = randomlist[x]
    Usrsim = randomlist[Imaxi]
    Usrrow = Mrandom_user.iloc[x]
    UsrMrat = Usrrow.max()
    UsrMid =  list(Usrrow).index(UsrMrat)+1
    UsrmovirT = df_movies['title'].loc[df_movies['movieId'] == UsrMid].values[0]
    sssim = Vmaxi
    list_data = [user,Usrsim,UsrmovirT,UsrMrat,sssim]
    suggestL.append(list_data)

suggest = pd.DataFrame(suggestL,columns = ['user','user_recommend','movie_title','rating','pearson_score'])
suggest  = suggest.set_index('user')
suggest_sort = suggest.sort_values(by = 'pearson_score', ascending=False)
suggest_sort.iloc[:,0:3] 


# In[141]:


#2.4.3 คนที่มีความชอบตรงข้ามกันที่สุด
unsuggestL =[]

for x in range(0,len(randomlist)):
    Vmaxi = 1
    Imaxi = 0
    for y in range(0,len(randomlist)):
        v = pearson_similarity.iloc[x,y]
        if v < Vmaxi and x !=y :
            Vmaxi = v
            Imaxi = y
            
    user = randomlist[x]
    Usrsim = randomlist[Imaxi]
    Usrrow = Mrandom_user.iloc[x]
    UsrMrat = Usrrow.max()
    UsrMid =  list(Usrrow).index(UsrMrat)+1
    UsrmovirT = df_movies['title'].loc[df_movies['movieId'] == UsrMid].values[0]
    sssim = Vmaxi
    list_data = [user,Usrsim,UsrmovirT,UsrMrat,sssim]
    unsuggestL.append(list_data)

unsuggest = pd.DataFrame(unsuggestL,columns = ['user','user_recommend','movie_title','rating','pearson_score'])
unsuggest  = unsuggest.set_index('user')
unsuggest_sort = unsuggest.sort_values(by = 'pearson_score', ascending=True)
unsuggest_sort.iloc[:,0:3] 


# In[142]:


#ตอนที่ 3:


# In[145]:


#3.1 คำนวณความคล้ายกันของ movie genre ของคู่ ‘movieId’ ใดๆ จากตาราง movie genre feature
#3.1.1
MovM = new_df.set_index('movieId')
MovM = MovM.iloc[:,1:]
MovM = MovM.drop(columns = 'year')

nUser= 50
randomlistM = []
for i in range(0,nUser):
    n = random.randint(1,MovM.shape[0]+1)
    randomlistM.append(n)

randomlistM.sort()

randomlistMId = []
for i in randomlistM:
    movieId = df_movies['movieId'].loc[i]
    randomlistMId.append(movieId)

RandM = MovM.loc[randomlistMId]


# In[146]:


#3.1.2 คำนวณโดยใช้ตัววัด cosine_similarity()

CSmovies = sklearn.metrics.pairwise.cosine_similarity(RandM)
CSmovies = pd.DataFrame(CSmovies,columns = randomlistMId )
CSmovies['movieId'] = randomlistMId
CSmovies = CSmovies.set_index('movieId')
CSmovies


# In[147]:


#3.1.3 คำนวณโดยใช้ตัววัด Pearson’s similarity

PSmovies = RandM.T.corr ( method ='pearson' )
PSmovies


# In[148]:


#3.2 


# In[149]:


#SPM = SPM
SPM= []
top = 5

for row in range(0,PSmovies.shape[0]):
    for column in range(row,PSmovies.shape[1]):
        tempc = [randomlistMId[row],randomlistMId[column],PSmovies.iloc[row,column],df_movies['title'].loc[df_movies['movieId'] == randomlistMId[row]].values[0],df_movies['title'].loc[df_movies['movieId'] == randomlistMId[column]].values[0]]
        if tempc[0] != tempc[1] :
            SPM.append(tempc)
         
column2 = ['movieId1','movieId2','pearson_similarity_score_movie','movie_title1','movie_title2']

sortPS = pd.DataFrame(SPM,columns = column2)

sortPS = sortPS.set_index(['movieId1','movieId2'])
sortPS_sort = sortPS.sort_values(by=['pearson_similarity_score_movie'], ascending=False)
sortPS_sort.head(top)


# In[150]:


sortPS_sort.tail(top)


# In[153]:


#3.2.3 รายการของ user ที่ให้ rating >= 3.0
M_pear =  []
M_num_pear = []
sort_U_rate = []

for i in range(0,5):
    M_pear.append([sortPS_sort.index.get_level_values(0)[i],sortPS_sort.index.get_level_values(1)[i]])

for i in range(0,len(M_pear)):
    M_num_pear.append([randomlistM[randomlistMId.index(M_pear[i][0])],randomlistM[randomlistMId.index(M_pear[i][1])]])

for i in range(0,len(M_pear)):
    for user in range(0,user_matrix.shape[0]):
        rate1 = user_matrix.iloc[user,M_num_pear[i][0]]
        #rate2 = user_matrix.iloc[user,M_num_pear[i][1]]
        if rate1 >= 3.0:
            UID = user + 1
            sort_U_rate_list = [UID,M_pear[i][0],rate1,df_movies['title'].loc[df_movies['movieId'] == M_pear[i][0]].values[0] , M_pear[i][1],df_movies['title'].loc[df_movies['movieId'] == M_pear[i][1]].values[0]]
            sort_U_rate.append(sort_U_rate_list )

sort_U_rate_df = pd.DataFrame(sort_U_rate,columns = ['userID','movieID1','rating','movie','movieID2','recommend movie'])
sort_U_rate_df


# In[ ]:





# In[ ]:




