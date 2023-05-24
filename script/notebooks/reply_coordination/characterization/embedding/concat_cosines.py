#### **This notebook concat cosine embeddings**

import pandas as pd
import numpy as np
import datetime
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import sys
import os

import importlib

#### packages
import helper.strategy_helper as st
import helper.visualization as viz_hp
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config_hp
import helper.pandas_helper as pd_hp


#### **Concat all cosines**
config = config_hp.config()
embedding_path = config['EMBEDDINGS_PATH']

missing_tweets = embedding_path['missing_tweets']

df_missing = pd.read_pickle(missing_tweets)
ids = df_missing['missing_tweetids'].tolist()

path = '/N/slate/potem/data/derived/reply_embedding/'

df_all = pd.DataFrame()
for id in ids:
    filename = f'replier_tweetid_combination_{id}*'
    files = os.path.join(path, filename)
    print(files)
    file_list = glob.glob(files)
    if len(file_list) == 0:
        continue
        
    df = pd.read_pickle(file_list[0])
    print('read_sucessfully')
    df_all = df_all.append(df, ignore_index=True)
    
### **Add tweet label id**
config = config_hp.config()
balanced = config['BALANCED']

balanced_neg_conversation = balanced['balanced_neg_conversation']
balanced_pos_conversation = balanced['balanced_pos_conversation']

df_neg = pd.read_pickle(balanced_neg_conversation)
df_pos = pd.read_pickle(balanced_pos_conversation)

df_conv = df_neg.append(df_pos)
grouped = df_conv.groupby(['poster_tweetid', 'tweet_label'])

# Get the keys of the groupby operation
keys = list(grouped.groups.keys())

# Create a DataFrame from the keys
df_grp = pd.DataFrame(data=keys, columns=['poster_tweetid','tweet_label'])

print(df_grp.head())


#### **Save the data**
config = config_hp.config()
embedding_path = config['EMBEDDINGS_PATH']
df_temp = df_all.merge(df_grp[['poster_tweetid',
                                'tweet_label'
                               ]])


cosine_pair_wise = embedding_path['cosine_pair_wise']
df_cosine_pair = pd.read_pickle(cosine_pair_wise)
df_cosine_all = df_cosine_pair.append(df_temp, 
                                      ignore_index=True)


all_cosine_pairwise = embedding_path['all_cosine_pairwise']
df_cosine_all.to_pickle(all_cosine_pairwise)