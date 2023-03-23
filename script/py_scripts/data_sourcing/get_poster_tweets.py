import pandas as pd
import numpy as np
import datetime
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from tqdm import tqdm
import sys
import os

import importlib

#### packages
import helper.strategy_helper as st
import helper.visualization as vz
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config
import helper.pandas_helper as pd_hp


def get_poster_tweets(config):
    '''
    Gets the poster tweets 
    '''
    
    config = config.config()
    path = config['PATHS']

    external_reply = path['external_reply']
    
    derived_path = path['derived_path']
    user_path = os.path.join(derived_path, 'posters')
    poster_path = os.path.join(user_path, 'posters_*')

    poster_tweet_path = file_hp.create_folder(derived_path, 
                                              'posters_tweets')
    for id_file in glob.glob(poster_path):
        parts = id_file.split(os.sep)
        index = parts[-1].split('_')[2]
        filename = parts[-1].split('.')[0]

        if int(index) <= 60:
            continue
            
        # if int(index) > 100:
        #     continue

        start_time = filename.replace('posters', 'start_time') + '.txt'
        end_time = filename.replace('posters', 'end_time') + '.txt'
        campaign_json = f'{filename}.jsonl'

        path_to_json = os.path.join(poster_tweet_path, campaign_json)
        start_time_file = os.path.join(user_path, start_time)
        end_time_file =  os.path.join(user_path, end_time)

        start_file = file_hp.read_file(start_time_file)
        end_file = file_hp.read_file(end_time_file)
        posters = file_hp.read_file(id_file)

        for i, user in enumerate(posters):
            job_info = filename.replace('posters_', '')
            poster_filename = f'{job_info}_{user}.jsonl'
            path = os.path.join(poster_tweet_path, poster_filename)

            start_file[i] = pd.to_datetime(start_file[i]) + pd.Timedelta(0, unit='s')
            end_file[i] = pd.to_datetime(end_file[i]) + pd.Timedelta(0, unit='s')

            start_file[i] = start_file[i].isoformat('T')
            end_file[i] = end_file[i].isoformat('T')

            command = f'twarc2 timeline --start-time={start_file[i]} ' \
            f'--sort-order=relevancy  --use-search --exclude-retweets --exclude-replies ' \
            f'--limit 1 {user} > {path}'

            os.system(command)
            
            print(command)

def get_poster_tweets_with_count(config):
    '''
    Gets the poster tweets
    :param config: config file to be loaded
    '''
    
    config = config.config()
    poster = config['POSTER_PATH']
    poster_alive_with_tweet_count_file = poster['poster_alive_with_tweet_count_file']
    
    poster_control = config['POSTER_CONTROL']
    poster_control_tweets = poster_control['poster_control_tweets']
    poster_tweet_path = file_hp.create_folder(poster_control_tweets, 
                                              'posters_new_tweets')
    poster_ids = file_hp.read_file(poster_alive_with_tweet_count_file)
    
    for row in poster_ids:
        row = row.strip('][').split(', ')

        user = int(row[0])
        start_time = row[1]
        count = row[2]
        poster_filename = f'control_tweets_{user}.jsonl'
        path = os.path.join(poster_tweet_path, poster_filename)

        start_time = pd.to_datetime(start_time) + pd.Timedelta(0, unit='s')
        start_time = start_time.isoformat('T')

        command = f'twarc2 timeline --start-time={start_time} ' \
        f' --sort-order=relevancy --use-search --exclude-retweets --exclude-replies ' \
        f'--limit {count} {user} > {path}'

        os.system(command)

# 
        
if __name__ == "__main__":
    # get_poster_tweets(config)
    get_poster_tweets_with_count(config)