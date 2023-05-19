#This script gets the 100 dimension instead of 728

import pandas as pd
import numpy as np
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import json
from tqdm import tqdm
import sys
import os

import importlib

#### packages
import helper.strategy_helper as st
import helper.visualization as vz
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config_hp
import helper.pandas_helper as pd_hp
import helper.twitter_helper as twitter_hp

import torch


#Load files
importlib.reload(config_hp)

config = config_hp.config()

embedding_path = config['EMBEDDINGS_PATH']
reply_multilanguage_embedding = embedding_path['reply_multilanguage_embedding']

df_emb = pd.read_pickle(reply_multilanguage_embedding)


X = torch.stack(df_emb['embeddings'].tolist())
Y = X[: ,:, :100] 

dim1_size = Y.size(0)

# Convert each 2-dimensional slice to a list
tensor_list = [Y[i, :, :] for i in range(dim1_size)]

df_100 = df_emb[['replier_tweetid', 
                 'poster_userid',
                 'poster_tweetid',
                 'replier_userid']]
df_100['embeddings'] = tensor_list

### Save the file
importlib.reload(config_hp)

config = config_hp.config()

embedding_path = config['EMBEDDINGS_PATH']
reply_multilanguage_embedding_100 = embedding_path['100_reply_multilanguage_embedding']

df_100.to_pickle(reply_multilanguage_embedding_100)
