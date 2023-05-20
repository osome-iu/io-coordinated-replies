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
import argparse
from sklearn.metrics.pairwise import cosine_similarity


import importlib

#### packages
#import helper.strategy_helper as st
#import helper.visualization as viz_hp
#import helper.helper as hp
#import helper.file_helper as file_hp
import config.config as config_hp
#import helper.pandas_helper as pd_hp
#import helper.twitter_helper as twitter_hp
#import helper.clean_tweet as clean_hp


#importlib.reload(config_hp)

#config = config_hp.config()
#balanced = config['BALANCED']

#positive_conv = balanced['balanced_pos_conversation']
#df_pos = pd.read_pickle(positive_conv)

#negative_conv = balanced['balanced_neg_conversation']
#df_neg = pd.read_pickle(negative_conv)

#df = df_pos.append(df_neg)

#get combination
from itertools import combinations

def find_combinations(group):
    combinations_list = list(combinations(group, 2))
    
    return combinations_list

def get_combination(df):
    df = df.sort_values(by=['poster_tweetid'])
    ids = df['poster_tweetid'].unique()[:10]
    
    importlib.reload(config_hp)

    config = config_hp.config()

    embedding_path = config['EMBEDDINGS_PATH']

    combination = embedding_path['combination']

    df_poster = df.loc[df['poster_tweetid'].isin(ids)][['poster_tweetid',
                                                        'replier_tweetid']]
    
    print(len(df_poster))
    
    df_comb = df_poster.groupby('poster_tweetid')['replier_tweetid'].apply(
list(combinations(group, 2))).reset_index()
    print('here')
    
    if os.path.exists(combination):
        df_already = pd.read_pickle(combination)
        df_combined = pd.concat([df_comb, df_already], 
                                ignore_index=True)
        print('Combined :', len(df_combined))
        
        df_combined.to_pickle(combination)
    else:
        df_comb.to_pickle(combination)
        
        print('Single :', len(df_comb))


from itertools import combinations

def try_get_combination(df, ids):
    print('Start the combination')
#     df_unq = df.groupby(['poster_tweetid', 
#                          'replier_userid'])['replier_tweetid'].last().reset_index()   
#     df_size = df_unq.groupby(['poster_tweetid'])[
#         'replier_tweetid'].nunique().to_frame('count').reset_index()
#     max_num = df_size['count'].max()
    
#     start = 4000
#     # end = start + 1000
#     ids = df_size.loc[(df_size['count'] >= start)]['poster_tweetid']
    
#     if len(ids) == 0:
#         return 0
    
    df = df.astype({
        'poster_tweetid': str
    })
    
    # df_poster = df.loc[df['poster_tweetid'].isin([ids])]
    # df_poster = df.head()

#     importlib.reload(config_hp)

#     config = config_hp.config()

#     embedding_path = config['EMBEDDINGS_PATH']

#     combination = embedding_path['combination_7']

    df_poster = df.loc[df['poster_tweetid'].isin([ids])][['poster_tweetid',
                                                        'replier_tweetid']]
    
    print(len(df_poster))
    
    print('starting')
    
    df_sample = df_poster.sample(n=1000)
    
    df_comb = df_sample.groupby('poster_tweetid')['replier_tweetid'].apply(lambda x:
        list(combinations(x, 2))).reset_index()
    
    print('list here')
    print(df_comb.info())

    df_exploded = df_comb.explode('replier_tweetid')

    print('Exploded number :', len(df_exploded))

    df_exploded['replier_x'] = df_exploded['replier_tweetid'].apply(
        lambda x: x[0])
    df_exploded['replier_y'] = df_exploded['replier_tweetid'].apply(
        lambda x: x[1])
    df_emb = df_exploded.merge(df[['replier_tweetid', 'embeddings']],
                  left_on='replier_x',
                  right_on='replier_tweetid'
                 )
    df_emb = df_emb.merge(df[['replier_tweetid', 'embeddings']],
                      left_on='replier_y',
                      right_on='replier_tweetid'
                     )
    print(df_emb.info())
    print('Embdding here')

    def get_cosine(df):
        '''
        Get the cosine similarity of the vector list
        :param vector_list: list of embedding vectors

        :return list
        '''

        df['cosine'] = df.apply(lambda x: round(
                cosine_similarity(x.embeddings_x, x.embeddings_y)[0][0],
                2),
            axis=1)

        return df
    
    
    df_emb_cosine = get_cosine(df_emb)
    
    folder = '/N/slate/potem/data/derived/reply_embedding'
    
    df_emb_cosine[['poster_tweetid', 'replier_x', 'replier_y', 'cosine']].to_pickle(
        f'{folder}/replier_tweetid_combination_{ids}.pkl.gz')
    
    print('here')
    
    return df_emb_cosine

def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(
        description='get cosine of embedding')
    
    parser.add_argument('--ids',
                        dest='ids',
                        help='Ids')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ids = args.ids
    
    #Load files
    print('Load embedding: start')
    importlib.reload(config_hp)

    config = config_hp.config()

    embedding_path = config['EMBEDDINGS_PATH']
    reply_multilanguage_embedding = embedding_path['reply_multilanguage_embedding']

    df_embedding = pd.read_pickle(reply_multilanguage_embedding)

    print('Load embedding: done')

    df_comb = try_get_combination(df_embedding, ids)
    
    
# python /N/slate/potem/project/infoOps-strategy/script/py_scripts/get_replier_tweetid_combination.py --ids=1000096002135781376 
