# In[142]:


#ตอนที่ 3:
#ต้องรันต่อจากตอนที่ 2

# In[145]:


#3.1 คำนวณความคล้ายกันของ movie genre ของคู่ ‘movieId’ ใดๆ จากตาราง movie genre feature
#3.1.1
MovM = new_df.set_index('movieId')
MovM = MovM.iloc[:,1:]
MovM = MovM.drop(columns = 'year')

nUser= 100
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