#### **This script tests different feature classification result**

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
import helper.stat_helper as stat_helper

#### Load files

import helper.stat_helper as stat_hp
importlib.reload(stat_hp)

config = config_hp.config()
stats = config['STATS']

df_all_stat = pd.read_pickle(stats['all_features'])


#### Run models

columns_not_include = [
    'list_quote_count',
    'list_like_count',
    'list_retweet_count',
    'list_reply_count',
    'list_reply_time_diff',
    # 'list_age_stats',
     'mad_diff_min',
    'cof_diff_min',
    'list_language_count',
    'list_mention_count',
    'list_hashtag_count',
    'list_url_count',
    'list_num_hashtags',
    'count_diff_min',
    'list_diff_min',
    'list_num_url',
    'list_cosine',
    'var_cosine',
    'var_diff_min',
    'var_mention_count',
    'var_num_hashtags',
    'reply_count_by_age',
    # 'org_reply_count'
]

print(len(columns_not_include))

# columns_not_include.extend(
#         ['poster_tweetid','tweet_label', 'replier_userid', 'replier_label'])
    
columns_to_keep = list(set(df_all_stat.columns) - set(columns_not_include))

diff_features = {'Reply Time': 'diff_min',
                 'Language Count': 'lang_count',
                 'Num of reply targeted tweet got': 'org_reply_count',
                 'Engagement metric': 'quote_count|like_count|retweet_count|reply_count',
                 'Reply similarity': 'cosine',
                 'Entites': 'mention_count|num_hashtags|num_url',
                 'All': 'diff_min|lang_count|org_reply_count|quote_count|like_count|retweet_count|reply_count|cosine|mention_count|num_hashtags|num_url'
                }

print('Running the model')

df_test = df_all_stat[columns_to_keep]

all_result = []
for feature in diff_features:
    feat = diff_features[feature] + '|tweet_label|poster_tweetid'
        
    df_filtered = df_test.filter(regex=feat, axis=1)
    
    total_col = len(df_filtered.columns)
    
    print(total_col)
    
    df_return, roc, prf_1, prf_0, mean_score, std_score = stat_hp.run_model(df_filtered,
                                  columns_not_include=columns_not_include,
                                  model_type='random', 
                                  pca=True,
                                  y_column = 'tweet_label',
                                  filename=None)
    
    all_result.append([feature, total_col, roc, prf_1[0], 
                       prf_1[1], prf_1[2],
                       prf_0[0], prf_0[1], prf_0[2],
                       mean_score, std_score
                      ])
    print(all_result)
    
(pd.DataFrame(data=all_result,
              columns=['feature', 'total_data',
                       'roc', 'precision_1', 'recall_1',
                       'f1_1', 'precision_0', 'recall_0',
                       'f1_0', 'mean_f1', 'std_f1']
             )
 
).to_pickle('./data/different_feature_result.pkl.gz')