import pandas as pd
import numpy as np
import time
#import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
#import data_layer as data_layer
import Recommenders as Recom
#import Evaluation as Evaluation

#-----------------------------------------------------------------------------------------------------------------

transactions = pd.read_csv(r"Consolidated.csv")

#-----------------------------------------------------------------------------------------------------------------
print(transactions.shape)
transactions.head(2)

#-----------------------------------------------------------------------------------------------------------------

transactions['Purchase'] = transactions['Purchase'].apply(lambda x: [str(i) for i in x.split('|')])
transactions.head(2).set_index('Users')['Purchase'].apply(pd.Series).reset_index()

#-----------------------------------------------------------------------------------------------------------------

pd.melt(transactions.head(2).set_index('Users')['Purchase'].apply(pd.Series).reset_index(), 
             id_vars=['Users'],
             value_name='Purchase') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['Users', 'Purchase']) \
    .agg({'Purchase': 'count'}) \
    .rename(columns={'Purchase': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'Purchase': 'product'})
    
#-----------------------------------------------------------------------------------------------------------------

s=time.time()

data = pd.melt(transactions.set_index('Users')['Purchase'].apply(pd.Series).reset_index(), 
             id_vars=['Users'],
             value_name='Purchase') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['Users', 'Purchase']) \
    .agg({'Purchase': 'count'}) \
    .rename(columns={'Purchase': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'Purchase': 'product'})
data['product'] = data['product']
print("Execution time:", round((time.time()-s)/60,2), "minutes")

#-----------------------------------------------------------------------------------------------------------------

data.sort_values('purchase_count',ascending = 0).head(3)

#-----------------------------------------------------------------------------------------------------------------

#The Popularity model takes the most popular items for recommendation.
#These items are products with the highest number of sells across customers.
#The Popularity model doesn’t give any personalizations, it only gives the same list of recommended items to every user.

#-----------------------------------------------------------------------------------------------------------------

users = data['Users'].unique()
#for i in range(len(users)):
#    print("["+str(i)+"]"+users[i])

train_data, test_data = train_test_split(data, test_size = 0.20, random_state=0)
#print(train_data.head(2))

#-----------------------------------------------------------------------------------------------------------------
 
pr = Recom.popularity_recommender_py()
pr.create(data, 'Users', 'product')  

#-----------------------------------------------------------------------------------------------------------------
print("----------------------------------------------------------------------")
print("!!!!!!!!!!!- POPULARITY-BASED RECOMMENDATION -!!!!!!!!!!")
print("----------------------------------------------------------------------")
user_id = users[181]
print(pr.recommend(user_id))

#-----------------------------------------------------------------------------------------------------------------

is_model = Recom.item_similarity_recommender_py()
is_model.create(data, 'Users', 'product')

#-----------------------------------------------------------------------------------------------------------------
print("----------------------------------------------------------------------")
print("!!!!!!!!!!!- PERSONALISED RECOMMENDATION -!!!!!!!!!!")
print("----------------------------------------------------------------------")
user_id = users[181]
user_items = is_model.get_user_items(user_id)
print("Training data product for the user userid: %s:" % user_id)

for user_item in user_items:
    print(user_item)
print("--------------------------------")
print("Recommendation process going on:")
print("--------------------------------")
print(is_model.recommend(user_id))

#----------------------------------------------------------------------------------------------------------------

#is_model.get_similar_items([''])