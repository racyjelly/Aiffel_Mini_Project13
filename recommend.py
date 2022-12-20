#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import implicit
import pandas as pd
import os

print(np.__version__)
print(scipy.__version__)
print(implicit.__version__)


# In[2]:


import os
rating_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/ratings.dat'
ratings_cols = ['user_id', 'movie_id', 'ratings', 'timestamp']
ratings = pd.read_csv(rating_file_path, sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
orginal_data_size = len(ratings)
ratings.head()


# In[3]:


# 3점 이상만 남깁니다.
ratings = ratings[ratings['ratings']>=3]
filtered_data_size = len(ratings)

print(f'orginal_data_size: {orginal_data_size}, filtered_data_size: {filtered_data_size}')
print(f'Ratio of Remaining Data is {filtered_data_size / orginal_data_size:.2%}')


# In[4]:


# ratings 컬럼의 이름을 counts로 바꿉니다.
ratings.rename(columns={'ratings':'counts'}, inplace=True)
ratings['counts'].head()


# In[5]:


# 영화 제목을 보기 위해 메타 데이터를 읽어옵니다.
movie_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/movies.dat'
cols = ['movie_id', 'title', 'genre'] 
movies = pd.read_csv(movie_file_path, sep='::', names=cols, engine='python', encoding='ISO-8859-1')
movies.head()


# ## 분석 시작

# In[6]:


using_col = ['user_id', 'movie_id', 'counts']
data = ratings[using_col]
data.head(5)


# 1. ratings에 있는 유니크한 영화 개수
# 2. ratings에 있는 유니크한 사용자 수
# 3. 가장 인기 있는 영화 30개(인기순)

# In[7]:


# 유저 수
print("User num: ", data['user_id'].nunique())
# 영화 수
print("Movie num: ", data['movie_id'].nunique())
# 평점은 어차피 3,4,5
print("Rating: ", data['counts'].nunique())


# In[8]:


import copy
data_new = data.copy()
data_new['view'] = data_new['movie_id'].value_counts()
data_new = data_new.sort_values(by=['view','counts'],  ascending=False)
# 가장 유저가 많이 본 무비 아이디 정렬해서 pop_movie에 저장 그리고 평점도
data_new


# In[9]:


pop_movie_df = pd.merge(data_new, movies, left_on="movie_id", right_on="movie_id", how="left")
pop_movie_df.head(30)


# In[10]:


pop_movie_df['title'] = pop_movie_df['title'].str.lower()
pop_movie_df['genre'] = pop_movie_df['genre'].str.lower()
pop_movie_df.head(5)


# In[11]:


pop_movie_df[pop_movie_df['title'].str.contains('lady and the')]
pop_movie_df[pop_movie_df['title'].str.contains('cinderella')]
pop_movie_df[pop_movie_df['title'].str.contains('snow')]

my_favor_id = [2080, 1022, 594]
my_favor_movie = []
for i in my_favor_id:
    condition = pop_movie_df.movie_id==i
    movie = pop_movie_df.loc[condition].title.iloc[0]
    my_favor_movie.append(movie)
my_playlist = pd.DataFrame({'user_id':['jennie97']*3, 'movie_id': my_favor_id, 'counts': [8]*3})

if not data.isin({'user_id':['jennie97']})['user_id'].any():
    data = data.append(my_playlist)

data.tail(8)
print(my_favor_movie)


# ## CSR matrix를 직접 만들어 봅시다.

# In[12]:


data = pd.merge(data, movies, left_on="movie_id", right_on="movie_id", how="left")
data['title'] = data['title'].str.lower()

# 고유한 유저, 영화를 찾아내는 코드
user_unique = data['user_id'].unique()
print(user_unique)
movie_unique = data['title'].unique()
print(movie_unique)


# In[13]:


user_to_idx = {v:k for k,v in enumerate(user_unique)}
movie_to_idx = {v:k for k,v in enumerate(movie_unique)}
movie_to_idx


# In[14]:


# user_to_idx.get을 통해 user_id 칼럼의 모든 값을 인덱싱한 series 구하기
temp_user_data = data['user_id'].map(user_to_idx.get).dropna()
if len(temp_user_data) == len(data):
    print("user_id column indexing is completed!")
    data['user_id'] = temp_user_data
else:
    print("user_id column indexing failed!")


# In[15]:


temp_movie_data = data['title'].map(movie_to_idx.get).dropna()
if len(temp_movie_data) == len(data):
    print('movie_id column indexing is completed!')
    data['movie_id'] = temp_movie_data
else:
    print('movie_id column indexing failed!')


# In[16]:


data.head(6)


# In[19]:


from scipy.sparse import csr_matrix

num_user = data['user_id'].nunique()
print(num_user)
num_movie = data['title'].unique()
print(num_movie)

csr_data = csr_matrix((data.counts, (data.user_id, data.movie_id)))
csr_data


# In[20]:


from implicit.als import AlternatingLeastSquares
import os
import numpy as np

# implicit 라이브러리에서 권장하고 있는 부분입니다. 학습 내용과는 무관합니다.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS']='1'


# In[21]:


als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False, iterations=20, dtype=np.float32)


# In[22]:


csr_data_t = csr_data.T
csr_data_t.shape


# In[23]:


als_model.fit(csr_data_t)


# In[25]:


jennie, movie = user_to_idx['jennie97'], movie_to_idx[my_favor_movie[0]]
jennie_vector, movie_vector = als_model.user_factors[jennie], als_model.item_factors[movie]
print("나의 선호도:", round(np.dot(jennie_vector, movie_vector),2))


# ## 내가 좋아하는 영화와 비슷한 영화 추천 받기

# In[26]:


want_movie = "10 things i hate about you (1999)"
want_movie_id = movie_to_idx[want_movie]
similar_movie = als_model.similar_items(want_movie_id, N=15)
similar_movie

idx_to_movie = {v:k for k,v in movie_to_idx.items()}
idx_to_movie


# In[37]:


def get_similar_movie(movie_name: str):
    movie_id = movie_to_idx[movie_name]
    similar_movie = als_model.similar_items(movie_id)
    recommend = []
    for i in similar_movie:
        recommend.append(idx_to_movie[i[0]])
    return print("What artists do I like? ===> ", recommend[1:3])


# In[38]:


get_similar_movie("10 things i hate about you (1999)")


# In[53]:


user = user_to_idx['jennie97']
movie_recommend = als_model.recommend(user, csr_data, N=20, filter_already_liked_items=True)
foru = []
for i in movie_recommend:
    foru.append(idx_to_movie[i[0]])
print("This is for U! ==>", foru[1:4])


# In[ ]:




