import pandas as pd
import numpy as np




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
        

def ratios(df):
    n_users = df['userid'].nunique()
    retweets = df.loc[df['is_retweet'] == True]
    n_retweets = len(retweets)
    n_tweets = len(df.loc[(df['is_retweet'] == False) & \
                (~df['quoted_tweet_tweetid'].notnull()) & \
                (df['in_reply_to_tweetid'].isnull())])
    n_total = len(df)
    n_replies = len(df_mexico.loc[~df_mexico['in_reply_to_tweetid'].isnull()])
    n_quoted = len(df_mexico.loc[~df_mexico['quoted_tweet_tweetid'].isnull()])

    
    print('\n ------- Retweet --------------\n')
    df_retweet = df.loc[df['is_retweet'] == True]
    df_retweet_grp = df_retweet.groupby(
        ['retweet_tweetid']).size().to_frame(
        'retweet_count').reset_index()
    
    print('\n Retweet to user ratio :', round(n_retweets/n_users, 2))
    print('\n Retweet to tweet ratio :', round(n_retweets/n_tweets, 2))
    
    
    print('\n ------- Replies -------------\n')
    print('\n Replies to user ratio :', round(n_replies/n_users))
    print('\n Replies to tweet ratio : ', round(n_replies/n_tweets))
    print('\n Maximum replies a tweet got :',
          max(df_mexico_tweet_grp['retweet_count']))
    
    
        

# def retweet(df):
    

    
# def replies_ratios(df):
    
    
# def retweet_coordination():
    
    
# def similarity_in_name():
    

    
    
# def hashtag_coordination():

    
# def mention_coordination():
    
    
# def account_creation_time():
    
    
    
# def language_use():
    
    
# def bot_or_not():
    
    
# def handle_switching()


# def automated_tweeting()


# def narratives():