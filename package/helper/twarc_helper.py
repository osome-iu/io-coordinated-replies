import os


def get_tweets(id_file, output_file):
    '''
    Get tweets from tweet id file
    :param id_file: file which contain tweet ids
    :param output_file: file which will save json file
    '''
    
    command = f'twarc2 hydrate {id_file} {output_file}'
    
    print(command)
    
    os.system(command) 
    
    
def get_profile_info(profile_id_file, 
                     profile_json_file):
    '''
    Gets the user meta data
    :param profile_id_file: Twitter account userid file
    :param profile_json_file: File path with file name for
    json of profiles
    '''
    
    command = f'twarc2 users {profile_id_file} {profile_json_file}'
    
    print(command)
    
    os.system(command)  
    
    
    
def get_tweet_of_user(start_time,
                 end_time,
                 userid,
                 output
                ):
    '''
    Gets tweet of user within the specified time
    :param start_time: start time
    :param end_time: end time
    :param userid: account id
    :param output: output json file
    '''
    
    start_time = pd.to_datetime(start_time) + pd.Timedelta(0, unit='s')
    end_time = pd.to_datetime(end_time) + pd.Timedelta(0, unit='s')

    start_time = start_time.isoformat('T')
    end_time = end_time.isoformat('T')

    command = f'twarc2 timeline --start-time={start_time} ' \
    f'--end-time={end_time}  --no-context-annotations  --use-search --exclude-retweets --exclude-replies ' \
    f'{userid} > {output}'

    print(command)
          
    os.system(command)  