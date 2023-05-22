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

path = '/N/slate/potem/data/derived/reply_embedding/'
filename = 'replier_tweetid_combination*'
files = os.path.join(path, filename)
df_all = pd.DataFrame()
for file in glob.glob(files):
    print(file)
    
    if file == '/N/slate/potem/data/derived/reply_embedding/replier_tweetid_combination.pkl.gz':
        continue
        
    df = pd.read_pickle(file)
    
    df_all = df_all.append(df, ignore_index=True)
    
#### **Add tweet lable id**
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

cosine_pair_wise = embedding_path['cosine_pair_wise']

df_temp = df_all.merge(df_grp[['poster_tweetid',
                                'tweet_label'
                               ]])

df_temp.to_pickle(cosine_pair_wise)