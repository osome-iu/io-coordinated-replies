import pandas as pd
import numpy as np
import warnings
import datetime
from tqdm import tqdm
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import networkx as nx
import os
from itertools import combinations


'''
Author: Manita Pote
'''

def pickle_open(filename):
    '''
    Opens pickle file
    
    :param filename: name of file
    '''
    with open(filename, 'rb') as file:
        open_file = pickle.load(file)
        
        return open_file
    
def create_folder(output_path, folder_name):
    '''
    Creates folder in output path if folder does not exists
    
    :param output_path: path where folder existence to be
    checked
    :param folder_name: name of folder
    '''
    
    path = os.path.join(output_path, folder_name)
    
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
        
    return path
        
        
def extract_mentions(df, filter_tweet=False, explode=False) -> pd.DataFrame():
    '''
    Extracts all the mentions from tweet text
    
    :param df: dataframe
    :param filter_tweet: filter out all the tweets that does not have
    mentions in tweets
    :param explode: make each mention a row
    
    :return pandas dataframe
    '''
    
    df['mentions_from_text'] = df.tweet_text.apply(
        lambda x: list(set(re.findall(r'@\S+', x))))
    
    df['mentions_from_text'] = df['mentions_from_text'].apply(
        lambda x: list(set(x)))
    
    if filter_tweet:
        df = df.loc[df['mentions_from_text'].map(len) != 0]
        
    if explode:
        df = df.explode('mentions_from_text')
    
    return df


        
def extract_hashtags(df, filter_tweet=False, explode=False) -> pd.DataFrame():
    '''
    Extracts all the hashtags from tweet text
    
    :param df: dataframe
    :param filter_tweet: filter out all the tweets that does not have
    mentions in tweets
    :param explode: make each hashtags a row
    
    :return pandas dataframe
    '''
    
    df['hashtags_from_text'] = df.tweet_text.apply(
        lambda x: list(set(re.findall(r'#\S+', x))))
    
    df['hashtags_from_text'] = df['hashtags_from_text'].apply(
        lambda x: list(set(x)))
    
    if filter_tweet:
        df = df.loc[df['hashtags_from_text'].map(len) != 0]
        
    if explode:
        df = df.explode('hashtags_from_text')
    
    return df


def filter_tweet_count(df,
                       retweet=False, 
                       filter_tweet=5)-> pd.DataFrame:
    '''
    Filter users based on filter_tweet threshold
    
    :param df: Pandas dataframe of tweets
    :param retweet: Flag to filter retweet
    :param filter_tweet: Threshold for tweet count to filter
    
    :return pandas dataframe
    '''
    
    if 'retweet_tweetid' in df.columns:
        column_name = 'retweet_tweetid'
    else:
        column_name = 'retweeted_tweetid'
        
    if retweet != None:
        df = df.loc[(df['is_retweet'] == retweet)]
        
    userids = (df
        .groupby(['userid'])['tweetid']
        .size()
        .to_frame('tweet_count')
        .reset_index()
        .query('tweet_count >= {}'.format(filter_tweet))
        )['userid'].tolist()
    
    
    df = df.loc[df['userid'].isin(userids)]
    
    return df.astype({
        'tweetid': 'string',
        'userid': 'string',
        column_name: 'string'
    })

def bin_tweets_in_time(df, time_bin=24, time_part='H') -> dict:
    '''
    Bins the tweets based on time_bin
    
    :param time_bin: time to bin the tweets with
    
    :return dictionary with end time as key and dataframe of tweets within time range \
    as values
    '''
    
    df['tweet_time'] = pd.to_datetime(df['tweet_time'])
   
    #Time bins
    if time_part == 'H':
        delta_time = datetime.timedelta(hours=time_bin)
    else:
        delta_time = datetime.timedelta(0,1)
        
    time_lists = (pd.DataFrame(columns=['NULL'],
                               index=pd.date_range(df['tweet_time'].min(),
                                                   df['tweet_time'].max() + delta_time,
                                                freq =f'{time_bin}{time_part}'))
                  .index
                  .tolist())
                  
    
    #Filter tweets in bins
    df_time = pd.DataFrame(columns=['userid', 
                            'tweetid', 
                            'time_interval'])
    
    for i in tqdm((range(1, len(time_lists)))):
        temp_df = df.loc[(time_lists[i-1] <= df['tweet_time']) & \
                           (df['tweet_time'] < time_lists[i])]

        if len(temp_df) <= 1:
            continue
            
        temp_df = temp_df[['userid', 'tweetid']]
        temp_df['time_interval'] = time_lists[i]
        df_time = pd.concat([df_time, temp_df])

    return df_time


def bin_time_in_time(df, time_bin=24, time_part='H') -> dict:
    '''
    Bins the tweets based on time_bin
    
    :param time_bin: time to bin the tweets with
    
    :return dictionary with end time as key and dataframe of tweets within time range \
    as values
    '''
    
    df['tweet_time'] = pd.to_datetime(df['tweet_time'])
   
    #Time bins
    if time_part == 'H':
        delta_time = datetime.timedelta(hours=time_bin)
    else:
        delta_time = datetime.timedelta(0,1)
        
    time_lists = (pd.DataFrame(columns=['NULL'],
                               index=pd.date_range(df['tweet_time'].min(),
                                                   df['tweet_time'].max() + delta_time,
                                                freq =f'{time_bin}{time_part}'))
                  .index
                  .tolist())
                  
    
    #Filter tweets in bins
    df_time = pd.DataFrame(columns=['userid', 
                            'tweetid', 
                            'time_interval'])
    
    for i in tqdm((range(1, len(time_lists)))):
        temp_df = df.loc[(time_lists[i-1] <= df['tweet_time']) & \
                           (df['tweet_time'] < time_lists[i])]

        if len(temp_df) <= 1:
            continue
            
        temp_df = temp_df[['userid']]
        temp_df['time_index'] = str(i)
        df_time = pd.concat([df_time, temp_df])

    return df_time

def convert_tweetids_to_string(df_time, field='tweetid',
                               filter_threshold=False,
                               threshold=None
                              ) -> pd.DataFrame:
    '''
    Converts the time binned tweets to single single dataframe and convert the tweets \
    to single document tweets for each users
    
    :param df_time: dictionary of time binned dataframes
    
    :return pandas dataframe
    '''

    df_time = df_time.astype({field: int})
    df_time = df_time.astype({field: str})
    
    df_time = (df_time
                   .groupby('userid')[field]
                   .apply(list)
                   .reset_index(name='tweet_ids'))
        
    if filter_threshold is not False:
        df_time = df_time.loc[df_time['tweet_ids'].map(len) > threshold]
    
    df_time['tweet_ids'] = df_time['tweet_ids'].apply(lambda x: ' '.join(x))
    
    return df_time
    
def calculate_tf_idf_vector(df) -> pd.DataFrame:
    '''
    Calculates the tfidf vector for document of tweetids
    
    :param df: dataframe
    
    :return pandas dataframe
    '''
    corpus = df['tweet_ids'].tolist()

    vectorizer = TfidfVectorizer(
        use_idf=True,
    )
    x = vectorizer.fit_transform(corpus)
    df['vector'] = list(x)
    print(df.info())
    
    return df


def create_user_projection(df, column_to_join='time_interval', 
                           userid=None):
    '''
    Creates user projection network from bi-partite network
    
    :param df: dataframe
    :param column_to_join: column to join the bi-partite
    network on
    :param userid: only considering userid
    '''
    
    
    df = df.drop_duplicates(subset=['userid', column_to_join])
    
    if userid is not None:
        df = df.loc[df['userid'].isin(userid)]
    
    # print(df.info())
    
    # x = df[column_to_join].apply(lambda x: len(x) == 0)
    # print(len(x))
    df = df[['userid', column_to_join]].merge(df[['userid', column_to_join]],
                                 on=column_to_join,
                                 suffixes=('_x', '_y')
                                )
    df = df.loc[df['userid_x'] != df['userid_y']]
    df = df.drop(columns=[column_to_join])
    df = df.drop_duplicates()
    
    print(df.info())
    
    return df
    
def calculate_cosine_similarity(df_network, df_vector):
    df_network = df_network.drop_duplicates()
    df_network = df_network.loc[
        df_network['userid_x'] != df_network['userid_y']]
    df_network = df_network.merge(df_vector[['userid','vector']],
                          left_on='userid_x', 
                          right_on='userid')
    
    
    print(df_network.info())
    
    if len(df_network) == 0:
        return df_network
    
    df_network = df_network.merge(df_vector[['userid','vector']],
                                  left_on='userid_y', 
                                  right_on='userid',
                                  suffixes=('_11', '_22'))
    
    df_network = df_network.drop(['userid_11', 'userid_22'], axis=1)
    df_network['cosine'] = tqdm(df_network.apply(
        lambda x: round(cosine_similarity(x['vector_11'].toarray(), 
                                          x['vector_22'].toarray())[0][0], 2),
        axis=1))
    
    df_network = df_network.drop(columns=['vector_11', 'vector_22'])
    print(df_network['cosine'].unique())
    df_network = df_network.loc[df_network['cosine'] != 0]
    print('\n after filtering \n')
    print(df_network.info())
    
    return df_network.rename(columns={'userid_x': 'source', 'userid_y': 'target'})

    
def create_graph(df_network,
                 output_path,
                 campaign_name,
                 source_column = 'source',
                 target_column = 'target',
                 weight_column=None, 
                 type_text=''):
    '''
    Creates a networkx graph
    :param df_network: dataframe which contain edge list
    :param output_path: path where graph to be saved
    :param campaign_name: part of filename
    :param source_column: column to be used as source node
    :param target_column: column to be used as target node
    :param weight_column: column to be used as weight
    :param type_text: text to give more info about the network creation
    also used as part of filename
    
    :return networkx graph
    '''
    if len(df_network) == 0:
        print(f'\n This data has no users using same {type_text}')
              
        return
    
    G = nx.from_pandas_edgelist(df_network, 
                                source_column, 
                                target_column, 
                                [weight_column])
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
              
    print(f'\n ------------- {type_text} network information ----------\n')
    print(nx.info(G))
    print('\n Number of connected components :', len(Gcc))          
    print('\n Information for largest connected component: ')
    print(nx.info(G0))
              
    nx.write_gml(G, f'{output_path}/{campaign_name}_{type_text}_network.gml') 
    
    return G
    
    
def read_data(input_path, filename):
    '''
    Read the data file from the given location
    
    :param input_path: Location to input file
    :param filename: Name of the file
    :return pandas dataframe
    '''
                        
    parts = filename.split('.')
                        
    if parts[-1] == 'csv':
        df = pd.read_csv(f'{input_path}/{filename}')
        
        print(df.info())
        
        return df
                        
    if parts[-1] == 'gz' and parts[-2] == 'pkl':
        df = pd.read_pickle(f'{input_path}/{filename}')
        
        if len(df) <= 1:
            print('The dataframe has just one column')
            
            raise Exception('Data insufficient')
        else:
            print(df.info())
            
            return df
    else:
    
        raise Exception(
            '--filename : Invalid file format. \n Only pkl.gz and csv accepted')
        
        
        
def extract_tweet_client(df) -> pd.DataFrame():
    '''
    Extracts unqiue userid and tweet_client_name from tweets
    
    :param df: dataframe
    
    :return dataframe
    '''
    uniques = list((df.groupby(['userid', 'tweet_client_name'])
                 .groups
                 .keys()
                ))
    df_unique = pd.DataFrame(data=uniques, 
                             columns=['userid', 'tweet_client_name'])
    
    len_uniques = len(df_unique['tweet_client_name'].unique())
    
    print('\n Number of unique client name :', len_uniques)
    
    return df_unique, len_uniques


def filter_reply_count(df, filter_tweet = 5):
    '''
    Filter out users with less than 5 reply to any tweets
    
    :param df: dataframe
    :param filter_tweet: threshold to filter outs
    
    :return Dataframe
    '''
    
    df = df.loc[~df['in_reply_to_userid'].isnull()]
    
    userids = (df
        .groupby(['userid'])['in_reply_to_userid']
        .size()
        .to_frame('tweet_count')
        .reset_index()
        .query('tweet_count >= {}'.format(filter_tweet))
        )['userid'].tolist()
    
    df = df.loc[df['userid'].isin(userids)]
    
    return df



def get_cosine(vector_list):
    '''
    Get the cosine similarity of the vector list
    :param vector_list: list of embedding vectors
    
    :return list
    '''
    ref_comb = combinations(vector_list, 2)
    all_cosine = []
    
    for ref in ref_comb:
        cosine_ref_ref = round(cosine_similarity([ref[0]],
                                                 [ref[1]])[0][0],
                               2)
        all_cosine.append(cosine_ref_ref)
