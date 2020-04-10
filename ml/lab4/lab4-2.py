# In[126]:

#2
#ต้องรันต่อจากตอนที่ 1

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

order_pearson_similarity = pd.DataFrame(result,columns =['pearson_similarity_score','userId1','userId2']).set_index(['userId1','userId2']) 
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
