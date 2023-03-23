import os
import pandas as pd
import argparse
import importlib
import helper.strategy_helper as st
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as configuration



# importlib.reload(configuration)

def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Parse zipped campaign data')
    
    parser.add_argument('--file',
                        dest='file',
                        help='conversation file')
    parser.add_argument('--destination-folder',
                        dest='destination_folder',
                        default=None,
                        help='Folder where conversation is to be extracted')
                        
    return parser.parse_args()


def main(args):
    #Load config files
    config = configuration.config()
    path = config['PATHS']
    derived_path = path['derived_path']
    all_tweet_data = path['all_tweet_data']
    conversation_path = path['conversation_path']
    file = args.file

    #location of conversation ids and location where
    #json to be extracted
    # file = '500_iran_202012_ops_conversation_ids.txt'
    conversation_file = os.path.join(conversation_path,
                                     args.file)

    file_hp.create_folder(derived_path, 'extracted_conversation')

    extraction_location = os.path.join(derived_path, 
                                       'extracted_conversation')
    file_hp.create_folder(extraction_location, 'extracted_reply_5')
    
    extraction_location_new = os.path.join(extraction_location, 
                                       'extracted_reply_5')

    #get conversation
    st.get_conversation(conversation_file, extraction_location_new)
    

def get_conversations(args):
    '''
    Gets the conversations
    :param args: arguments passed
    '''
    conversation_file = args.file
    extraction_location_new = args.destination_folder
    
    st.get_conversation(conversation_file,
                        extraction_location_new)
    

if __name__ == "__main__":
    args = parse_args()
    
    # main(args)
    get_conversations(args)