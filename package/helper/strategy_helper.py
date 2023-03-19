import pandas as pd
import numpy as np
import warnings
import datetime
import gzip
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import difflib
import json
import os
import sys
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon

import config.config as config
# sys.path.insert(0, '/N/u/potem/Quartz/project/infoOps-strategy/script/helper')
# from visualization import *

config = config.config()
path = config['PATHS']
derived_path = path['derived_path']

userid = 'userid'
time_year = 'tweet_time_year' #tweet time in YYYY-MM-DD
count = 'count'
tweetid ='tweetid'
tweet_language = 'tweet_language'
in_reply_to_tweetid = 'in_reply_to_tweetid'

def read_ops_control_data(ops_file_path, control_file_path, 
                          includes=['ops', 'control']) -> dict:
    '''
    Read influence operation and control data
    
    :param ops_file_path: path to influence operation folder
    :param control_file_path: path to control file
    :param includes: list of string to decide what to include, 
    ops means include influence operation, control means include control dataset
    
    '''
    data = {}
    
    if 'ops' in includes:
        df_ops = pd.read_pickle(ops_file_path)
        data['ops'] = df_ops
        
    if 'control' in includes:
        if os.path.isfile(control_file_path) == False:
            data['control'] = []
        else:
            df_control = pd.read_pickle(control_file_path)
            data['control'] = df_control

    return data

def add_YYYY_MM_DD(df):
    df['tweet_time'] = pd.to_datetime(df['tweet_time'])
    df['tweet_time_year'] = df.tweet_time.map(
                lambda x: x.strftime('%Y-%m-%d'))
    
    return df

def add_tweet_time_month(df):
    df['tweet_time'] = pd.to_datetime(df['tweet_time'])
    df['tweet_time_year'] = df.tweet_time.map(
                lambda x: x.strftime('%Y-%m-%d'))

    df['tweet_time_month'] = pd.to_datetime(
                    df['tweet_time']).dt.to_period('M')
    
    return df


def get_date_range(min_date, max_date, column='tweet_time_month'):
    date_range = pd.date_range(min_date, max_date, 
                               freq='M', inclusive='both').tolist()
    date_range = [i.strftime('%Y-%m') for i in date_range]

    df = pd.DataFrame(data = date_range, columns=[column])

    df[column] = df[column].astype(str)
    
    return df



def add_date_range(df, df_date, column='tweet_time_month') -> pd.DataFrame():
    '''
    Adds date range in dataframe
    
    :param df: dataframe in which date range to be added
    :param df_date: dataframe which has date range in YYYY-MM format
    :param column: column name to merge two dataframe
    '''
    df[column] = df[column].astype(str)
    df = df.merge(df_date, on=column, how='outer')
    
    return df.fillna(0)

def check_2400_tweets(df, userid_column = 'userid', 
                      time_column= 'tweet_time_year', 
                      count_column='count',
                      tweet_coulmn='tweetid',
                      threshold=2400):
    df_groups = (df.groupby([userid_column, time_column])[tweet_coulmn]
                 .size()
                 .to_frame(count_column)
                 .reset_index()
                 .query(f'{count_column} > {threshold}')
                )
    
    print(f'Number of users who post more than {threshold}: ', 
          len(df_groups))
    
    
def get_above_and_below_row(df, condition_to_filter, 
                            sort_by='tweet_time') -> pd.DataFrame():
    '''
    Sorts the rows by some column and gets the above and below row of
    condition matched row
    
    :param df: dataframe
    :param condition_to_filter: condition to filter rows by
    :param sort_by: column to sort the dataframe with
    
    :return dataframe
    '''
    df_temp = df.sort_values(by=[sort_by], ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    m = (condition_to_filter)
    df_new = df_temp[m.shift(-1)|m.shift()|m]
    
    return df_new

def time_difference_between_tweets(df, column='tweet_time', 
                                   difference_type=None):
    
    df = df.drop_duplicates(subset=['tweetid'])
    df[column] = pd.to_datetime(df[column])
    df = df.sort_values(by=[column])
    
    df['time_shift'] = df.groupby(['userid'])[column].shift(periods=1)
    
    if difference_type == 'sec':
        df['delay'] = (df[column] - df['time_shift']).dt.total_seconds()

    if difference_type == 'days':
        df['delay'] = (df[column] - df['time_shift']).dt.days
        
    # df = df.loc[~df['delay'].isnull()]
    
    # df = df.dropna()
        
    return df


def filter_tweets(df, is_retweet, only_replies, threshold=10) -> pd.DataFrame():
    '''
    Filter the tweets either retweet, count or replies
    
    :param is_retweet: Boolen value to include retweet or not
    :param only_replies: Boolean value to include replies only
    :param threshold: Number of counts to filter the data by
    
    :return pandas dataframe
    '''
    
    df = df.loc[(df['is_retweet'] == is_retweet)]
        
    if only_replies:
        df = df.loc[~df['in_reply_to_tweetid'].isnull()]
        
    if (only_replies == False):
        df = df.loc[df['in_reply_to_tweetid'].isnull()]

    #filter out all the users with less than 10 tweets
    if threshold is not None:
        userids = (df.groupby(['userid'])['tweetid']
                   .size()
                   .to_frame('count')
                   .reset_index()
                   .query(f'count > {threshold}')
                  )['userid']

        df = df.loc[df['userid'].isin(userids)]
        
   
    
    return df


def parse_control_data(campaigns, input_file_path, 
                       output_file_path):
    total_row = []
    data = {}
    
    # for campaign in campaigns[year]:
        
        #path to control file
        # file_path = f'{path}/all_tweets/{year}/{campaign}/DriversControl'
        # filename = f'{file_path}/control_driver_tweets.jsonl.gz'
        
    print(f'\n----------- {campaign} : Control data parse start ---------- \n')
    if os.path.isfile(input_file_path) == False:
        return [campaigns, False]
    
    with gzip.open(input_file_path, 'r') as json_file:
        for row in json_file:
            one_row = json.loads(row)

            new_row = {
                'account_creation_date': one_row['user']['created_at'],
                'user_display_name': one_row['user']['name'],
                'userid': one_row['user']['id'],
                'user_screen_name': one_row['user']['screen_name'],
                'user_profile_url': one_row['user']['url'],
                'user_profile_description': one_row['user']['description'],
                'follower_count': one_row['user']['followers_count'],
                'following_count': one_row['user']['friends_count'],
                'user_reported_location': 0,

                'tweetid': one_row['id'],
                'tweet_text': one_row['full_text'],
                'tweet_language': one_row['lang'],
                'geo': one_row['geo'],
                'place': one_row['place'],
                'in_reply_to_tweetid': one_row['in_reply_to_status_id'],
                'in_reply_to_screen_name': one_row['in_reply_to_screen_name'],
                'in_reply_to_userid': one_row['in_reply_to_user_id'],
                'tweet_time': one_row['created_at'],
                'user_mentions': one_row['entities']['user_mentions'],
                'hashtags': one_row['entities']['hashtags'],
                'symbols': one_row['entities']['symbols'],
                'urls': one_row['entities']['urls'],
                'tweet_client_name': one_row['source'],
                'is_retweet': False,
                'retweeted_tweetid': 0,
                'retweeted_user_id': 0
            }

            if 'retweeted_status' in one_row:
                new_row['is_retweet'] = True
                new_row['retweeted_tweetid'] = one_row['retweeted_status']['id']
                new_row['retweeted_user_id'] = one_row['retweeted_status']['user']['id']


            if 'location' in one_row['user']:
                new_row['user_reported_location'] = one_row['user']['location']

            total_row.append(new_row)

    df= pd.DataFrame.from_records(data=total_row)
    df['tweet_time'] = pd.to_datetime(df['tweet_time'])
    df['tweet_time_year'] = df.tweet_time.map(
            lambda x: x.strftime('%Y-%m-%d'))

    df.to_pickle(
        f'{output_file_path}/{campaign}_tweets_control.pkl.gz')

    print(f'\n----------- {campaign} : Control data parse end ---------- \n')
            
            
            
def read_username_data(ops_file, control_file):
    for ops_file in glob.glob(ops_file):
        df_ops = pd.read_csv(ops_file)
        
    df_control = pd.read_csv(control_file)

    data = {
            'ops': df_ops,
            'control': df_control
            }
    
    return data



def handle_sharing(df, save_path, filename):
    '''
    Checks if the data has any user with more than one user screen name
    '''
    df = df.drop_duplicates()
    
    #@screen name
    df_username = (df
                .groupby(['userid'])['user_screen_name']
                .apply(lambda x: list(np.unique(x)))
                .to_frame('user_screen_name')
                .reset_index())
    
    df_username['count_names'] = df_username['user_screen_name'].map(len)
    
    print('Number of users with more than one screen name',
          len(df_username.loc[df_username['count_names'] > 1]))
    
    
def check_tweet_time_glitch(df, userid='userid'):
    userids = df.loc[df['delay'] == 0]['userid'].tolist()
    df = df.loc[df['userid'].isin(userids)]
    df_groups = df.groupby(['userid'])
    all_df = pd.DataFrame()
    
    for group in df_groups:
        df_user = group[1]
        df_user = df_user.sort_values(by=['tweet_time'], 
                                      ascending=True)
        df_user = df_user.reset_index(drop=True)
        m = (df_user['delay'] == 0)
        df_row = df_user[m.shift(-1)|m]
        
        df_text = (df_row[['tweet_time', 'time_shift', 
                           'delay', 'tweetid', 'tweet_text']]
                   .sort_values(by=['tweet_time'], 
                                ascending=True))
        
        df_tex_grp = (df_text
                    .groupby(['tweet_text'])
                    .size()
                    .to_frame('count')
                    .reset_index()
                   )
        
        df_text_new = df_text.merge(df_tex_grp, 
                                on='tweet_text', 
                                how='outer'
                               )
        
        df_user_text = (df_text_new.loc[
            (df_text_new['delay'] == 0) & \
            (df_text_new['tweet_text'].isin(
                df_tex_grp['tweet_text'])) &\
            (df_text_new['count'] == 1)])
    
        all_df = all_df.append(df_user_text, ignore_index=True)
        
    return all_df    



def per_day_activity_per_user(df, time_column, tweetid_column,
                              userid_column, new_column='mean'
                             ):
    df_grp = (df.groupby([time_column, userid_column])[tweetid_column]
                               .size()
                               .to_frame('count')
                               .reset_index())
                
    df_grp = (df_grp.groupby([time_column])['count']
                   .mean()
                   .to_frame(new_column)
                   .reset_index())
    
    max_date = max(df_grp[time_column])
    min_date = min(df_grp[time_column])
    df_range = get_date_range_day(min_date, max_date, 'tweet_time_year')
    df_grp = add_date_range(df_grp, df_range, 'tweet_time_year')

    return df_grp


def statistics(x, y):
    D_c =  1.36*1/len(x)*2**0.5
    alpha = 0.05
    
    

    print('KS test')
    print('Two-sided: The null hypothesis is that the two distributionsn are identical,\n F(x)=G(x) for all x; the alternative is that they are not identical.')
    print('\n')
    print('Significance level 0.05 and 0.01') 
    print('D_c,0.05 = ',D_c)
    print('D_c,0.01 = ', 1.628*1/len(x)*2**0.5)

    stat, p = ks_2samp(x, y)

    print('Statistics={}, p={}'.format(stat, p))

    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    print('\n\n')
    print('Mannwhitney U Test')

    stat, p = mannwhitneyu(x,y)

    print('Statistics={}, p={}'.format(stat, p))

    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

#     print('\n\n')
#     print('Wilcoxon Signed-Rank Test')

#     stat, p = wilcoxon(x, y)

#     print('Statistics={}, p={}'.format(stat, p))

#     if p > alpha:
#         print('Same distribution (fail to reject H0)')
#     else:
#         print('Different distribution (reject H0)')
        
        
        
        
def IQR(data):
    '''
    Gets interquartile range of data
    :param data: data of which we want IQR
    
    :return low_lim: lower limit
    :return up_lim: upper limits
    '''
    Q1 = np.percentile(data, 25, interpolation = 'midpoint') 
    Q2 = np.percentile(data, 50, interpolation = 'midpoint') 
    Q3 = np.percentile(data, 75, interpolation = 'midpoint') 

    IQR = Q3 - Q1 

    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    
    return low_lim, up_lim



def flatten_list(df, column_name):
    all_x = []
    temp_names = list(map(list, df[column_name]))
    
    for x in temp_names:
        all_x.append(x[0])
        all_x.append(x[1])
        
        
    return list(set(all_x))




# all_plot_path = plot_path

# df_describe = pd.read_pickle(f'{save_path}/all_handle_dump_stats.pkl.gz')

# print(df_describe.head())


# columns = ['outlier_frac', 'mean', 'median_more_Q3',
#            'mean_5', 'median_5']
# x_labels = ['Fraction of outliers (>IQR)',
#             'Mean number of mentioned accounts \n in outliers',
#             'Median number of mentioned accounts \n in outliers',
#             'Mean number of mentioned accounts \n in top 5%',
#             'Median number of mentioned accounts \n in top 5%',
#          ]

# file_path = f'{save_path}/all_handle_dump_stats.pkl.gz'

# parameters = {
#             'labels': '',
#             'size': (10,8),
#             'label_name': ['IO', 'Control'],
#             'x': [],
#             'reports': reports,
#             'diff_color': ['cuba_082020', 'iran_202012'],
#             'fontsize': 14,
#             'title': 'Handle dump',
#             'xlabel': 'Campaigns',
#             'ylabel': '',
#             'legend_location': 'upper right',
#             'save': {
#                 'path': plot_path,
#                 'filename': ''
#             },
#             'random_color': False
#             }
    
# for i, column in enumerate(columns):
#     sort_column = column
#     parameters['ylabel'] = x_labels[i]
#     parameters['save']['filename'] = f'multiple_bar_hand_dump_{sort_column}.png'
    
#     multi_bar_with_column(file_path,
#                           sort_column,
#                           plot_path,
#                           parameters
#                          )

def multi_bar_with_column(file_path,
                          sort_column,
                          plot_path,
                          parameters
                         ):
    df = pd.read_pickle(f'{file_path}')

    df = df.sort_values(by=[sort_column], 
                        ascending=False) #.head(10)

    df[sort_column] = df[sort_column].apply(
        lambda x: round(x, 2))

    df_pivot = df.pivot(index='campaign', 
                            columns='type', 
                             values=sort_column).reset_index()

    df_pivot = df_pivot.sort_values(by=['ops'], 
                                    ascending=False)

    labels = df_pivot['campaign']
    ops = df_pivot['ops']
    control = df_pivot['control']

    all_plot_path = plot_path
    
    parameters['labels'] = labels
    parameters['x'] = [ops, control]

    multiple_bar_graph(parameters)
    
    
def read_image_file(path, 
                    save_path, 
                    campaign,
                    type_of,
                   ):
    '''
    Reads all the image data
    :param path: path of the files
    :param campaign: name of campaign
    :param type_of: type of image file (eg: profile, banner)
    
    :return list
    '''
    source_path = os.path.join(path, f'*{type_of}')
    all_data = []
    
    for image in glob.glob(source_path):
        parts = image.split(os.sep)[-1]
        userid = parts.split('_')[0]
        
        all_data.append(userid)
        
    save_filename = os.path.join(save_path, f'{campaign}.pkl.gz')
    
    (pd.DataFrame(data=all_data, 
                 columns=['userid']
                 )
    ).to_pickle(save_filename)
    

# all_data
def search_campaign_file(campaigns, search_term,
                         includes=['ops', 'control']
                        ):
    '''
    Search the campaign from dictionary of campaigns and get the tweet data
    :param search_term: campaign to search
    :param includes: tweet data to include either control or influence operation
    
    :return list of dataframe
    '''
    for year in campaigns:
        for new_campaign in campaigns[year]:
            if new_campaign != search_term:
                continue
                
            ops_file_path = os.path.join(path, 
                                         'all_tweets',
                                         year,
                                         new_campaign,
                                         f'{new_campaign}_tweets.pkl.gz')
            control_file_path = os.path.join(path, 
                                             'all_tweets', 
                                             year, 
                                             new_campaign, 
                                             'DriversControl', 
                                             f'{new_campaign}_control.pkl.gz')

            data = read_ops_control_data(ops_file_path, 
                                         control_file_path,
                                         includes
                                        )
            return data
    

    
def similarity_in_display_name(df, 
                               fields=['username', 'mentioned_name'],
                               include_in_df=False
                              ):
    '''
    Gets similarity in string using difflib
    
    :param df: dataframe
    :param fields: column names to lookinto
    :param include_in_df: boolean to specify if the ratio column is included or
    not and to either return dataframe or list of ratios
    
    :return list or dataframe
    '''
    ratios = []
    
    for index, row in df.iterrows():
        name_1 = row[fields[0]]
        name_2 = row[fields[1]]
                
        ratio = round(difflib.SequenceMatcher(None, 
                                              name_1,
                                              name_2).ratio(), 2)
        ratios.append(ratio)
        
    if include_in_df == True:
        df['ratio'] = ratios
        
        return df
        
    return ratios



def count_reply_external(df):
    '''
    Return the count of replies done to external userid
    
    :param df: dataframe
    '''
    
    df = df.fillna(0)

    df_replies = df.loc[~(df['in_reply_to_tweetid'] == 0)]

    df_replies = df.loc[~(df['in_reply_to_userid'].isin(df['userid']))]
                       
    df_replies_grp = (df_replies.groupby(['userid'])['tweetid']
                      .size()
                      .to_frame('count_replies')
                      .reset_index())
    
    return df_replies_grp



def unique_user_involved_in_reply(df_ops, column_name='count'):
    '''
    Gets the count of users involved in replying to tweets
    
    :param df_ops: dataframe
    :param column_name: name of the column that has count data

    :return Dataframe
    '''
    df_only_reply = df_ops.loc[~df_ops['in_reply_to_tweetid'].isnull()]
    df_only_reply = df_only_reply.loc[df_only_reply['in_reply_to_tweetid'] != 0]
    df_only_reply = reply_to_external_users(df_only_reply)

    #Groupby the reply_to_tweetid
    df_reply_dist = (df_only_reply
                     .groupby(['in_reply_to_tweetid'])['userid']
                     .apply(lambda x: list(set(x)))
                     .to_frame('userid_list')
                     .reset_index()
                    )

    df_reply_dist['count'] = df_reply_dist['userid_list'].apply(
        lambda x: len(x))

    print('Maximum reply a tweet got :', df_reply_dist['count'].max())
    
    return df_reply_dist


def reply_to_external_users(dataframe,
                            language='all'
                           ) -> pd.DataFrame:
    '''
    Return the reply tweets to external userid
    
    :param dataframe: dataframe
    :param language: language of tweets default consider
    all language
    
    :return dataframe
    '''
    
    df = dataframe
    
    if language != 'all':
        df = df.loc[df[tweet_language] == language]
        
    df = df.fillna(0)
    df_replies = df.loc[~(df[in_reply_to_tweetid] == 0)]

    df_replies = df_replies.loc[~(df_replies[in_reply_to_tweetid].isin(df[userid]))]
    
    return df_replies



def bundle_campaign(all_campaigns=None,
                    bundle=None
                   ):
    '''
    Splits the campaign data into parts according to bundle list
    
    :param all_campaigns: dictionary of campaigns
    :param bundle: list of keys to split
    
    :return splited dictionary, concated names of keys
    '''
    
    if all_campaigns == None:
        all_campaigns = {
            '2021_12': ['CNHU_0621', 'CNCC_0621', 'MX_0621', 
                        'REA_0621', 'RNA_0621', 'Tanzania_0621', 
                        'uganda_0621', 'Venezuela_0621'],
            '2020_12': ['armenia_202012', 'GRU_202012', 'IRA_202012', 
                        'iran_202012'],
            '2020_09': ['ira_092020', 'iran_092020', 'thailand_092020'],
            '2020_08': ['cuba_082020', 'qatar_082020'], #control is present upto this point

            '2020_05' : [
                         'china_052020', 
                         'turkey_052020',
                         'russia_052020' #incomplete control data
                        ],
            '2020_04' : ['egypt_022020', 'honduras_022020',
                         'indonesia_022020', 'sa_eg_ae_022020',
                         'serbia_022020'],
            '2020_03': ['ghana_nigeria_032020'], #*
            '2019_11': ['saudi_arabia_112019'],
            '2019_08': ['china_082019',
                        'ecuador_082019', #*
                        'egypt_uae_082019', #*
                        'saudi_arabia_082019_1', 
                        'spain_082019', #*
                        'uae_082019'], #*
            '2019_06': ['catalonia_201906_1',
                        'russia_201906_1', #need checking
                        'iran_201906', #*
                        'venezuela_201906_1'
                       ],
            '2019_01': ['iran_201901_1', #*
                        'russia_201901_1', #*
                        'bangladesh_201901_1', #*
                        'venezuela_201901', #*
                       ],
            '2018_10': ['ira', #*
                        'iranian'],
        }


    if bundle == None:
        bundle = [
            ['2021_12', '2020_12', '2020_09', '2020_08'], 
            ['2020_05', '2020_04', '2020_03', '2019_11'], 
            ['2019_08', '2019_06', '2019_01', '2018_10']
        ]
    
    names = []
    splits = []

    for row in bundle:
        split_bundle = {key: all_campaigns[key] for key in row}
        
        splits.append(split_bundle)
        names.append(','.join(row))
    
    return splits, names


def parse_conversation_json(conversation_file, 
                            output_file=None) -> pd.DataFrame:
    '''
    Parse the conversation json file
    
    :param conversation_file: location (along with file name)
    to the conversation json file
    :param output_file: location (along with file name) to save
    the parsed file
    
    :return DataFrame
    '''
    
    all_replies = []
    
    with open(conversation_file, 'r') as json_file:
        for row in json_file:
            one_row = json.loads(row)

            for reply in one_row['data']:
                replies = {}

                replies['conversation_id'] = reply['conversation_id']
                replies['reply_text'] = reply['text']
                replies['tweet_id'] = reply['id']
                replies['lang'] = reply['lang']
                replies['in_reply_to_user_id'] = reply['in_reply_to_user_id']
                replies['created_at'] = reply['created_at']

                all_replies.append(replies)
    
    df = pd.DataFrame.from_records(data=all_replies)
    
    if output_file is not None:
        df.to_pickle(output_file)
        
    return df

def get_conversation(conversation_file,
                     extraction_location
                    ):
    '''
    Get the conversation from conversation ids (conversation ids
    are in txt file. Each line is one conversation id
    
    :param conversation_file: location where conversation file is
    located
    :param extraction_location: location where extracted conversation is
    saved
    '''
    for id_file in glob.glob(conversation_file):
        conversation_id_file = id_file.split(os.sep)[-1]
        campaign = conversation_id_file.split('.')[0]

        print(f'\n\n ---- Starting for campaign : {campaign} -- \n')

        campaign_json = f'{campaign}.jsonl'
        path_to_json = os.path.join(extraction_location, campaign_json)
        command = f'twarc2 conversations --archive {id_file} > {path_to_json}'

        os.system(command)  

        print(f'\n ---- Ending for campaign : {campaign} ------- \n')
        
        
        
def check_if_common_user(df_ops, df_control):
    '''
    Checks if there are common replied to tweetid and userid
    
    :param df_ops: First Dataframe
    :param df_control: Second Dataframe
    '''
    
    df_ops = df_ops.loc[~df_ops['in_reply_to_tweetid'].isnull()]
    df_control = df_control.loc[~df_control['in_reply_to_tweetid'].isnull()]
    
    df_ops = df_ops.astype({
        'in_reply_to_tweetid': int,
        'userid': int,
        'in_reply_to_userid': int
    })
    
    df_control = df_control.astype({
        'in_reply_to_tweetid': int,
        'userid': int,
        'in_reply_to_userid': int
    })
    
    df_merge_replied = df_ops.merge(
        df_control[['userid', 'in_reply_to_tweetid']], 
        on='in_reply_to_tweetid')
    
    print('common in_reply_to_tweetid ', len(df_merge_replied))
    
    df_merge_userid = df_ops.merge(
        df_control[['userid', 'in_reply_to_userid']], 
        on='in_reply_to_userid')
    
    print('Common users ', len(df_merge_userid))
    
    
    
def extract_mentions_from_user_mentions(df, explode=True):
    '''
    Extracts mentions from user_mentions column in which the 
    list of mentioned userids are in string form
    
    :param df: DataFrame
    :param explode: Boolean to specify whether to explode the list
    '''
    
    df_mentions = df.loc[~df['user_mentions'].isnull()]

    df_mentions['user_mentions'] = df_mentions['user_mentions'].apply(
        lambda x: x.strip('][')
    )

    df_mentions = df_mentions.loc[
        df_mentions['user_mentions'].map(len) != 0]

    df_mentions['user_mentions'] = df_mentions['user_mentions'].apply(
        lambda x: x.split(', ')
    )

    df_mentions = df_mentions.explode('user_mentions')
    df_mentions = df_mentions.astype({
        'user_mentions': int
    })

    if explode == True:
        df_mentions = df_mentions.loc[df_mentions['user_mentions'] != 0]
        
    return df_mentions


def get_original_tweet(df):
    '''
    Gets only original tweet
    
    :param df: Dataframe
    '''
    flag_reply = df['in_reply_to_tweetid'].isnull()
    flag_quoted = df['quoted_tweet_tweetid'].isnull()
    flag_retweet = df['is_retweet'] == False

    df_orignal_tweet = df.loc[flag_reply & flag_quoted & flag_retweet]

    return df_orignal_tweet



def extract_mentions_from_control(df, explode=True):
    '''
    Extracts the user mentions from dictionary of user mentions
    
    :param df: DataFrame
    :param explode: Blooean to whether to explode the user mentions
    list
    
    :return DataFrame
    '''
    
    df['user_mentions'] = df['user_mentions'].apply(
        lambda x: [y['id'] for y in x]
    )
    
    df = df.loc[
        df['user_mentions'].map(len) != 0]
    
    df = df.explode('user_mentions')
    
    df = df.astype({
        'user_mentions': int
    })
    
    return df


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



def get_data_path(path, 
             year, 
             campaign,
            ):
    '''
    Gets path to control and IO tweet file
    
    :param path: path to the directory where tweet is present
    :param year: year of compaign
    :param campaign: name of campaign
    :param check_control: whether to check if control is present
    
    return list
    '''
    
    ops_file_path = os.path.join(path, 
                                 year, 
                                 campaign, 
                                 f'{campaign}_tweets.pkl.gz')
    control_file_path = os.path.join(path, 
                                     year, 
                                     campaign, 
                                     'DriversControl', 
                                     f'{campaign}_control.pkl.gz')
    
    if os.path.isfile(control_file_path) == False:
        control_file_path = os.path.join(path, 
                                     year, 
                                     campaign, 
                                     'DriversControl', 
                                     f'{campaign}_control.pkl.gz')
        
    ops_flag = os.path.isfile(ops_file_path) == True
    
    if ops_flag:
        return {'ops': ops_file_path,
                'control': control_file_path
               }
    
    print('Files not found')
    
    return None
