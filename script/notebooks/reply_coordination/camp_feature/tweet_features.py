#### **This script gets the camp features**

import pandas as pd
import numpy as np
import warnings
import os

import importlib

#### packages
import helper.strategy_helper as st_hp
import config.config as config_hp
import helper.stat_helper as stat_hp


####################### Load files ###########################

config = config_hp.config()

balanced_pos = config['BALANCED']['balanced_pos_conversation']
balanced_neg = config['BALANCED']['balanced_neg_conversation']

df_pos = pd.read_pickle(balanced_pos)
df_neg = pd.read_pickle(balanced_neg)

##############################################################
df_no_null = df_pos.loc[~df_pos['campaign'].isnull()]

df_no_null['poster_tweetid'].nunique()

df_grp_name = (df_no_null
              .groupby(['poster_tweetid'])['campaign']
               .first()
               .reset_index()
              )
df_new = df_pos.merge(df_grp_name,
                      on='poster_tweetid',
                     )

df_new.drop(columns=['campaign_x'], inplace=True)
df_new.rename(columns={'campaign_y': 'campaign'}, inplace=True)

######################### Divide data ########################

import random

def get_camp_data(df_pos, df_neg, campaign):
    if campaign == 'remain':
        top_5 =  df_grp_camp['campaign'].head().tolist()
        df_pos_camp = df_pos.loc[~df_pos['campaign'].isin(top_5)]
    else:
        df_pos_camp = df_pos.loc[df_pos['campaign'] == campaign]
    
    #How many to get from the negative dataset
    
    #find number of tweets of each user in positive
    df_pos_grp = (df_pos_camp
                  .groupby('poster_userid')['poster_tweetid']
                  .nunique()
                  .to_frame('count_pos')
                  .reset_index()
                 )
    
    print(df_pos_camp['poster_tweetid'].nunique())
    
    #Get users in negative
    df_neg_camp = df_neg.loc[df_neg['poster_userid'].isin(
        df_pos_camp['poster_userid']
    )]
    
    #find out how many tweets in negative set
    df_neg_grp = (df_neg_camp
                  .groupby('poster_userid')['poster_tweetid']
                  .nunique()
                  .to_frame('count_neg')
                  .reset_index()
                 )
    
    #Find least of all
    df_grp = df_pos_grp.merge(df_neg_grp,
                              on='poster_userid',
                             )
    df_grp['min_count'] = df_grp[['count_pos', 'count_neg']].min(axis=1)
   
    #sample least in negative dataset
    df_neg_common = df_neg.merge(df_grp,
                              on='poster_userid'
                             )
    
    #Balance the data in negative
    sampled_df = df_neg_common.groupby(['poster_userid'], group_keys=False).apply(
        lambda group: group.loc[group['poster_tweetid'].isin(
            random.sample(list(set(group['poster_tweetid'])),
                          list(set(group['min_count']))[0]
                         )
        )]
    ).reset_index(drop=True)
    
    sampled_df.drop(columns=['min_count', 
                             'count_pos',
                             'count_neg',
                             'common'
                            ],
                    inplace=True)

    df_pos_camp.drop('common',
                     inplace=True,
                     axis=1
                    )
    
    #Get rest of the data
    df_rest_pos = df_pos.loc[
        ~df_pos['poster_tweetid'].isin(df_pos_camp['poster_tweetid'])
    ]
    
    df_rest_neg = df_neg.loc[~df_neg['poster_tweetid'].isin(sampled_df['poster_tweetid'])]
    
    df_rest = pd.concat([df_rest_pos, df_rest_neg],
                        ignore_index=True
                       )
    
    df_camp = pd.concat([df_pos_camp, sampled_df],
                        ignore_index=True
                       )
    
    return df_camp, df_rest


################### campaigns ##############################

df_grp_camp = (df_new
               .groupby(['campaign'])['poster_tweetid']
               .nunique(dropna=False)
               .to_frame('count')
               .reset_index()
               .sort_values(by=['count'],
                            ascending=False
                           )
               
              )


###################### Features Functions #####################################
import datetime
import re
import helper.helper as hp

def metric(df_camp=None, stats=None):
    '''
    Get metric features and summary statistics
    :param df_camp: Campaign dataframe
    :param stats: List to hold dataframe for different metrics
    
    :return list of dataframe
    '''
    #metric (like, quote, reply, retweet)
    config = config_hp.config()

    metric_pos = config['METRIC']['io_tweet_metric_label']
    metric_neg = config['METRIC']['pc_tweet_metric_label']
    
    df_pos_metric = pd.read_pickle(metric_pos)
    df_neg_metric = pd.read_pickle(metric_neg)
    
    df_all_metric = pd.concat([df_pos_metric,
                               df_neg_metric
                              ])
    df_all_metric['poster_tweetid'] = df_all_metric['poster_tweetid'].astype(str)
    df_camp['poster_tweetid'] = df_camp['poster_tweetid'].astype(str)
    
    df_got = df_all_metric.loc[
        df_all_metric['poster_tweetid'].isin(df_camp['poster_tweetid'])
    ]
    
    print(df_camp['poster_tweetid'].nunique())
    print(df_got['poster_tweetid'].nunique())
    
    for x in ['retweet_count', 'reply_count', 'quote_count', 'like_count']:
        df_got_1 = df_got.loc[df_got['tweet_label'] == 1]
        df_got_0 = df_got.loc[df_got['tweet_label'] == 0]
        
        stat_hp.KS_test(df_got_1[x], df_got_0[x])
    
        df_stat_men = stat_hp.all_stat(df_got,
                                       column_to_groupby='poster_tweetid',
                                       column_to_take=x,
                                       label='tweet_label'
                                      )
        stats.append(df_stat_men)
        
        
    return stats
        

def reply_time_diff(df_camp=None):
    '''
    Calculates the summary statistics of reply time difference
    :param df_camp: Campaign dataframe
    
    :return dataframe
    '''
    #### get original IO tweets
    df_camp['conversation_id'] = df_camp['conversation_id'].astype(int)
    df_camp['created_at'] = df_camp['conversation_id'].apply(
        lambda x: hp.get_created_date(x)
    )
    df_camp['tweet_time'] = df_camp['tweet_time'].apply(
        lambda t: t.replace(tzinfo=None)
    )
    df_camp['diff'] = df_camp['tweet_time'] - df_camp['created_at']

    df_camp['diff'] = df_camp['diff'] + datetime.timedelta(seconds=1)

    df_camp['diff'] = df_camp['diff'].apply(
        lambda x: int(np.ceil((x.total_seconds() + 1) / 3600))
    )
    
    df_diff_pos = df_camp.loc[df_camp['tweet_label'] == 1]
    df_diff_neg = df_camp.loc[df_camp['tweet_label'] == 0]
    
    stat_hp.KS_test(df_diff_pos['diff'], 
                df_diff_neg['diff']
               )
    
    df_stat = stat_hp.all_stat(df_camp,
                                   column_to_groupby='poster_tweetid',
                                   column_to_take='diff',
                                   label='tweet_label'
                                  )
    
    print(df_stat.head())
    
    
    return df_stat
    
def lang_count(df_camp=None):
    '''
    Gets langage count of tweets
    
    :param df_camp: Campaign Dataframe
    
    :return dataframe
    '''
    config = config_hp.config()
    language = config['STATS']['language_count']
    
    df = pd.read_pickle(language)
    
    df_lang = df.loc[df['poster_tweetid'].isin(
        df_camp['poster_tweetid']
    )]
    
    print(df_camp['poster_tweetid'].nunique())
    print(df_lang['poster_tweetid'].nunique())
    
    print(df_lang.columns)
    
    return df_lang[['poster_tweetid', 'lang_count', 'tweet_label']]


def entity(df_camp=None, stats=None):
    '''
    Calculates the summary statistics
    :param df_camp: Campaign Dataframe
    :param stats: List to store dataframe of different entity
    summary statistics
    
    :return list of dataframe
    '''
    df_camp['num_mentions'] = df_camp['tweet_text'].apply(
        lambda x: list(set(re.findall(r'@\S+', x)))
    )
    df_camp['num_mentions']  = df_camp['num_mentions'].apply(
        lambda x: len(x)
    )
    df_camp['num_hashtags'] = df_camp['tweet_text'].apply(
        lambda x: list(set(re.findall(r'\B\#(\w+)', x)))
    )
    df_camp['num_hashtags']  = df_camp['num_hashtags'].apply(
        lambda x: len(x)
    )
    df_camp['num_url'] = df_camp['tweet_text'].apply(
        lambda x: list(set(re.findall(r'https?://\S+|www\.S+', x)))
    )  
    df_camp['num_url']  = df_camp['num_url'].apply(
        lambda x: len(x)
    )
    
    print(df_camp.head())
    
    for x in ['num_mentions', 'num_hashtags', 'num_url']:
        stat_hp.KS_test(df_camp[x], 
                        df_camp[x]
                       )
        df_stat = stat_hp.all_stat(df_camp,
                                       column_to_groupby='poster_tweetid',
                                       column_to_take=x,
                                       label='tweet_label'
                                      )
        
        stats.append(df_stat)
        
    return stats

def load_cosine():
    '''
    Load cosine files
    
    :return Dataframe of positive and negative class
    '''
    config = config_hp.config()
    pos_cosine = config['EMBEDDINGS_PATH']['pos_cosine_with_replier_info']
    neg_cosine = config['EMBEDDINGS_PATH']['neg_cosine_with_replier_info']
    
    df_pos = pd.read_pickle(pos_cosine)
    # df_neg = pd.read_pickle(neg_cosine)
    
    df_pos['poster_tweetid'] = df_pos['poster_tweetid'].astype(str)
    # df_neg['poster_tweetid'] = df_neg['poster_tweetid'].astype(str)
    
    df_neg = []

    return df_pos, df_neg

def cosine(df_camp=None, 
           campaign=None,
           df_pos=None, 
           df_neg=None
          ):
    '''
    Calculates the summary statistics of the cosine similarity
    :param df_camp: Dataframe of campaign
    :param stats: List of dataframe
    :param df_pos: Dataframe of positive class that has cosine
    :param df_neg: Dataframe of negative class that has cosine
    
    :return dataframe-
    '''
    config = config_hp.config()
    camp_cosine = config['CAMP_FEAT']['camp_features']
    campaign_cosine = os.path.join(camp_cosine, 
                                   f'{campaign}_cosine_pos.pkl.gz'
                                  )
    
    if os.path.exists(campaign_cosine):
        df = pd.read_pickle(campaign_cosine)
        
        return df
    
    
    df_camp['poster_tweetid'] = df_camp['poster_tweetid'].astype(str)
    
    df_camp_pos = df_pos.loc[
        df_pos['poster_tweetid'].isin(
            df_camp['poster_tweetid']
        )
    ]
    df_camp_neg = df_neg.loc[
        df_neg['poster_tweetid'].isin(
            df_camp['poster_tweetid']
        )
    ]

    print(df_camp_pos['poster_tweetid'].nunique())
    print(df_camp_neg['poster_tweetid'].nunique())

    stat_hp.KS_test(df_camp_pos['cosine'], 
                    df_camp_neg['cosine']
                   )
    
    df_all = pd.concat([df_camp_pos,df_camp_neg],
                       ignore_index=True
                      )
    
    df_stat = stat_hp.all_stat(df_camp_pos,
                            column_to_groupby='poster_tweetid',
                            column_to_take='cosine',
                            label='tweet_label'
                           )
    
    
    
    df_stat.to_pickle(f'{campaign_cosine}')
    
    print('file saved')
    
    return df_stat

def get_org_tweet_reply(df_camp=None):
    '''
    Gets the original reply count of targeted tweets
    
    :param df_camp: Campaign dataframe
    
    :return Dataframe
    '''
    config = config_hp.config()
    org_reply = config['STATS']['original_tweet_reply_count']
    
    df_reply = pd.read_pickle(org_reply)
    
    df_found = df_reply.loc[
        df_reply['poster_tweetid'].isin(df_camp['poster_tweetid'])
    ]
    
    print(df_found['poster_tweetid'].nunique())
    print(df_camp['poster_tweetid'].nunique())
    
    return df_found[['poster_tweetid', 'org_reply_count', 'tweet_label']]
                                                                   
    
def get_statistics(df_camp=None, campagin_name=None, 
                   df_pos=None, df_neg=None):
    '''
    Gets the features for the campaign data
    '''
    print(campagin_name)
    
    config = config_hp.config()
    camp_feat = config['CAMP_FEAT']['camp_features']
    campaign_feat = os.path.join(camp_feat, 
                                 f'{campagin_name}_features.pkl.gz'
                                )
    print(campaign_feat)
    
    if os.path.exists(campaign_feat):
        return
    
    #metric (like, quote, reply, retweet)
    stats = []
    
    stats = metric(df_camp, stats)
    
    print('** Metric done **')
    
    #time_diff
    
    df_stat = reply_time_diff(df_camp)
    
    stats.append(df_stat)
    
    print('** Time diff done **')
    
    #lang_count
    df_stat = lang_count(df_camp)
    
    stats.append(df_stat)
    
    print('** Lang count done **')
    
    #entity doing this
    #mention
    #hashtag
    #url
    stats = entity(df_camp, stats)
    
    print('** Entity done **')
    
    #cosine
    df_stat = cosine(df_camp,
                     campagin_name,
                     df_pos,
                     df_neg
                    )
    
    stats.append(df_stat)
    
    print('** Cosine done **')
    
    df_stat = get_org_tweet_reply(df_camp)
    
    stats.append(df_stat)
    
    print('**  org reply cout **')
    
    df = stats[0]
    for x in stats[1:]:
        df = df.merge(x,
                      on=['poster_tweetid', 'tweet_label']
                     )
        
    df.to_pickle(f'{campaign_feat}')
    
    print('** All features saved **')
    
    
##### Load cosine ##############################

df_pos_cosine, df_neg_cosine = load_cosine()


################ create features ################
camp_list = df_grp_camp['campaign'].head().tolist()
camp_list.extend('remain')

for campaign in ['remain']:
    print('Campaign ', campaign) 
    
    df_camp, df_rest = get_camp_data(df_new, df_neg, campaign)
    
    print(df_camp['poster_tweetid'].nunique())
    # df_pos = []
    # df_neg = []
    
    get_statistics(df_camp, campaign, df_pos_cosine, df_neg_cosine)
    
#     rest = f'{campaign}_rest'
    
#     df_stat = cosine(df_rest,
#                      rest,
#                      df_pos_cosine,
#                      df_neg_cosine
#                     )
    
    
    # get_statistics(df_rest, 
    #                rest, 
    #                df_pos=df_pos_cosine, 
    #                df_neg=df_neg_cosine
    #               )    

