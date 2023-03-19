import pandas as pd
import numpy as np
import orjson as orjson
from tqdm import tqdm
import os
import argparse
import glob
from zipfile import ZipFile
import gzip

def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Parse zipped campaign data')
    
    parser.add_argument('--input',
                        dest='input_path',
                        help='Input file path')
    
    parser.add_argument('--output',
                        dest='output_path',
                        help='Output file path')
    
    parser.add_argument('--campaign-name',
                        dest='campaign_name',
                        help='Name of campaign')
    
    parser.add_argument('--filename',
                        dest='tweet_filename',
                        help='Tweet file name')
                        
    return parser.parse_args()
    

def parse_data(args):
    '''
    Parse the zip file in input file location and saves at output file location
    :param args: arguments passed in commandline
    '''
    extension = args.tweet_filename.split('.')
    parts = args.campaign_name.split('_')
    years = parts[-1]
    df_all = pd.DataFrame()
    
    if len(parts) > 3 and parts[2].isnumeric():
        years = years + '_' + parts[2]
        
    create_folder(f'{args.output_path}', years)
    
    # new_path = os.path.join(f'{args.output_path}', years)
    
    # create_folder(new_path, args.campaign_name)
    
    if extension[-1] == 'gz' and extension[0] != 'all':
        # create_folder(f'{args.output_path}/{year}', args.campaign_name)
        
        with gzip.open(f'{args.input_path}/{args.tweet_filename}', 
               'rb') as f_in:
            df = pd.read_csv(f_in)
            
            df.to_pickle(
                f'{args.output_path}/{years}/{args.campaign_name}_tweets.pkl.gz'
            )
            
        return
       
    if extension[0] == 'all':
        args.tweet_filename = '*.csv.gz'
        
        for filename in glob.glob(f'{args.input_path}/{args.tweet_filename}'):
            print(filename)
            with gzip.open(f'{filename}', 
               'rb') as f_in:
                df = pd.read_csv(f_in)
                df_all = pd.concat([df_all, df], axis=0)
                print(len(df_all))
        
        df_all.to_pickle(
        f'{args.output_path}/{years}/{args.campaign_name}_tweets.pkl.gz')
        
        return
    
    for filename in glob.glob(f'{args.input_path}/{args.tweet_filename}'):
        with ZipFile(filename, 'r') as zip_archive:
            for item in zip_archive.filelist:
                print(item.filename)
                if '._' in item.filename:
                    continue
                if '__' in item.filename:
                    continue
                    
                type_extension = item.filename.split('.')[-1]
                
                if type_extension == 'csv':
                    df = pd.read_csv(zip_archive.open(item.filename))
                    # break

                file_parts = item.filename.split('/')

                if len(file_parts) > 1:
                    if file_parts[-1] != '':
                        year = (file_parts[-1].split('.')[0]).split('_')[-1]
                        print(year)
                        
                        create_folder(
                            f'{args.output_path}/{years}/{args.campaign_name}', year)
                    else:
                        print(file_parts)
                        continue
                        
                # full_path = '/'.join(file_parts)
                # df = pd.read_csv(zip_archive.open(full_path))
                
                # df.to_pickle(
                #     f'{args.output_path}/{years}/{args.campaign_name}/{year}/{year}_tweets.pkl.gz')
                
                df_all = pd.concat([df_all, df], axis=0)
                
                
    df_all.to_pickle( f'{args.output_path}/{years}/{args.campaign_name}/{args.campaign_name}_tweets.pkl.gz')
    df_all.to_csv(f'{args.output_path}/{years}/{args.campaign_name}/{args.campaign_name}_tweets.csv.gz',
                  compression='gzip', index=False)
    

def create_folder(output_path, campaign_name):
    '''
    Creates campaign folder in output path if file does not exists
    
    :param args: arguments passed in command line
    '''
    isExist = os.path.exists(f'{output_path}/{campaign_name}')

    if not isExist:
        os.makedirs(f'{output_path}/{campaign_name}')
    
def main(args):
    # create_folder(args.output_path, args.campaign_name)
    parse_data(args)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
# python parse-zip-data.py --input=/N/slate/potem/YYYY_MM/2019_08/saudi_arabia_082019_1 --output=/N/slate/potem/data/derived/all_tweets --campaign-name=saudi_arabia_082019_1 --filename=unhashed_2019_08_saudi_arabia_082019_1_saudi_arabia_082019_1_tweets_csv_unhashed.zip


# ops --task=create_driver_control_ds --bearer-token='' --tweets-path='/N/slate/potem/data/derived/all_tweets/2021_12/uganda_0621/' uganda_0621_all_tweet.csv.gz
