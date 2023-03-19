import pandas as pd
import numpy as np
import os
import argparse

#### packages
import helper.strategy_helper as st
import helper.visualization as vz
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config
import helper.pandas_helper as pd_hp
import helper.embedding as emb_hp


def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Parse zipped campaign data')

    parser.add_argument('--type-of',
                        dest='type_of',
                        help='Type of data to get for embedding eg: poster or reply')
    
    return parser.parse_args()


def get_reply_data(path):
    '''
    Read replies tweet
    :param path: path where data file is
    
    :return dataframe
    '''
    df_replies = pd.read_pickle(path)
    df_eng = df_replies.loc[df_replies['tweet_language'] == 'en']
    df_eng = df_eng.drop_duplicates()

    df_eng = df_eng.drop_duplicates(subset=['replier_tweetid',
                                            'tweet_text'], 
                                    keep='last')
    return df_eng

def get_poster_data(path):
    '''
    Gets poster tweet data
    :param path: path where data is
    
    :return dataframe
    '''
    
    df_tweets = pd.read_pickle(path)
    df_eng = df_tweets.loc[df_tweets['lang'] == 'en']
    df_eng = df_eng.drop_duplicates()

    df_eng = df_eng.drop_duplicates(subset=['tweetid',
                                            'text'], 
                                    keep='last')
    return df_eng

def get_embedding(df,
                  text_column,
                  save_path,
                  column_to_keep=['replier_tweetid']):
    '''
    Gets the embedding of text column
    :param df: Dataframe which has data
    :param text_column: Tweet text column
    :param save_path: Path where the embedding to be saved
    :param column_to_keep: Columns to be added after getting embedding
    '''
    
    sentences = df[text_column].tolist()
    
    #Get sentence embedding and save
    embeddings = emb_hp.get_sentence_embedding(sentences)

    column_to_keep = column_to_keep + ['embeddings']
    
    df_emb = pd.DataFrame(columns=column_to_keep)
    
    df_emb['embeddings'] = embeddings
    
    for column in column_to_keep:
        df_emb[column] = df_eng[column].tolist()

    df_emb.to_pickle(f'{save_path}')

    
def get_config(config, 
               type_of='poster'):
    '''
    Get the path of files
    :param type_of: Type to determine kind of file to get
    '''
    config = config.config()
    
    if type_of == 'poster':
        path = config['POSTER_PATH']
        input_file = path['parsed_poster_org_tweets']
        output_file = path['poster_org_tweet_embedding']
        
    if type_of == 'reply':
        path = config['PATHS']
        input_file = path['all_replies']
        output_file = path['reply_embedding']
    
    return input_file, output_file

if __name__ == "__main__":
    args = parse_args()
    input_file, output_file = get_config(config, args.type_of)
    
    if args.type_of == 'poster':
        df = get_poster_data(input_file)
        text_column = 'text'
        column_to_keep=['tweetid', 'author_id']
    if args.type_of == 'reply':
        df = get_reply_data(input_file)
        text_column = 'tweet_text'
        column_to_keep=['replier_tweetid', 'poster_tweetid']
        
    get_embeddings(df,
                  text_column,
                  output_file,
                  column_to_keep)