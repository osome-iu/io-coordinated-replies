import pandas as pd
import json
import re
import glob

def get_empty_profile_dict():
    '''
    Creates an empty variable to store profile information
    '''
    profiles = {
        'created_at': None,
        'verified': None,
        'description': None,
        'protected': None,
        'username': None,
        'id': None,
        'proile_image_url': None,
        'pinned_tweet_id': None,
        'name': None,
        'public_metrics': None,
        'followers_count': None,
        'following_count': None,
        'tweet_count': None,
        'listed_count': None,
        'location': None,
        'parameter': None
    }
    
    return profiles


def set_values_in_profile_dict(values):
    '''
    Sets variable values for profiles
    :param values: values from which values are to be stored
    
    :return dictionary
    '''
    profile = get_empty_profile_dict()
    profile['created_at']=values['created_at']
    profile['verified']=values['verified']
    profile['description']=values['description']
    profile['protected']=values['protected']
    profile['username']=values['username']
    profile['id']=values['id']
    profile['profile_image_url']=values['profile_image_url']

    if 'location' in values:
        profile['location'] = values['location']
    if 'pinned_tweet_id' in values:
        profile['pinned_tweet_id']=values['pinned_tweet_id']

    profile['name']=values['name']

    if 'public_metrics' in values:
        public_metrics = values['public_metrics']
        profile['followers_count']=public_metrics['followers_count']
        profile['following_count']=public_metrics['following_count']
        profile['tweet_count']=public_metrics['tweet_count']

        if 'listed_count' in public_metrics:
            profile['listed_count']=public_metrics['listed_count']
        
    return profile



def set_values_for_profile_error(values):
    '''
    Sets variable values for profiles which are not available
    :param values: values to be stored
    
    :return dictionary
    '''
    if values['resource_type'] != 'user':
        return None
    
    profile = get_empty_profile_dict()
    
    if 'parameter' in values:
        profile['parameter'] = values['parameter']
    
    if 'suspended' in values['detail']:
        profile['verified'] = 'suspended'
    if 'not find user' in values['detail']:
        profile['verified'] = 'not found'
        
    profile['description'] = values['detail']
    profile['id'] = values['value']
    
    return profile


def parse_profile_json(profile_file, 
                       output_file=None) -> pd.DataFrame:
    '''
    Parse the profile json file
    
    :param profile_file: location (along with file name)
    to the profile json file
    :param output_file: location (along with file name) to save
    the parsed file
    
    :return DataFrame
    '''
    
    all_profiles = []
    
    with open(profile_file, 'r') as json_file:
        for row in json_file:
            one_row = json.loads(row)
            
            if 'errors' in one_row:
                for single_profile in one_row['errors']:
                    profile = set_values_for_profile_error(single_profile)
                    
                    if profile != None:
                        all_profiles.append(profile)

            if 'data' not in one_row:
                continue
                
            for single_profile in one_row['data']:
                profile = set_values_in_profile_dict(single_profile)
                
                if profile != None:
                    all_profiles.append(profile)
                
    
    df = pd.DataFrame.from_records(data=all_profiles)
    
    if output_file is not None:
        df.to_pickle(output_file)
        
    return df


def get_empty_tweet_dict():
    '''
    Gets the empty tweet dictionary
    
    :return dictionary
    '''
    
    tweet = {
        'text': None,
        'conversation_id': None,
        'lang': None,
        'entities': None,
        'possibly_sensitive': None,
        'reply_settings': None,
        'created_at': None,
        'edit_history_tweet_ids': None,
        'tweetid': None,
        'author_id': None,
        'retweet_count': None,
        'reply_count': None,
        'like_count': None,
        'quote_count': None,
        'impression_count': None,
        'expanded_url': None,
        'display_url': None,
        'in_reply_to_user_id': None,
        'referenced_tweets': None,
        'context_annotations': None,
        'entity_annotations': None,
        'cashtags': None,
        'hashtags': None,
        'mentions': None,
    }
    
    return tweet


def add_entity_values(tweet, values):
    '''
    Parse the entities in tweet
    
    :param tweet: New tweet object which hold values
    :param values: Return object from twitter API
    
    :return dictionary
    '''

    if 'entities' not in values:
        return tweet
    
    if 'urls' in values['entities']:
        urls = values['entities']['urls']
        expanded_url = []
        display_url = []

        for url in urls:
            expanded_url.append(url['expanded_url'])
            display_url.append(url['display_url'])

        tweet['expanded_url'] = expanded_url
        tweet['display_url'] = display_url
    
    if 'annotations' in values['entities']:
        annotations = values['entities']['annotations']
        all_ann = []
        for ann in annotations:
            all_ann.append({'type': ann['type'], 
                            'probability': ann['probability']
                           })
        tweet['entity_annotations'] = all_ann
        
    if 'cashtags' in values['entities']:
        cashtags = values['entities']['cashtags']
        all_ann = []
        for ann in cashtags:
            all_ann.append(ann['tag'])
            
        tweet['cashtags'] = all_ann
        
    if 'hashtags' in values['entities']:
        hashtags = values['entities']['hashtags']
        all_ann = []
        for ann in hashtags:
            all_ann.append(ann['tag'])
            
        tweet['hashtags'] = all_ann
    
    if 'mentions' in values['entities']:
        mentions = values['entities']['mentions']
        all_ann = []
        for ann in mentions:
            all_ann.append([ann['username'], ann['id']])
            
        tweet['mentions'] = all_ann
        
        
    return tweet


def add_public_metric(tweet, values):
    '''
    Adds the public metric values to tweet 
    :param tweet: New tweet object
    :param values: Return object from Twitter API
    
    :return dictionary
    '''
    
    if 'public_metrics' not in values:
        return tweet
    
    metric = values['public_metrics']

    if 'retweet_count' in metric:
        tweet['retweet_count'] = metric['retweet_count']
    if 'reply_count' in metric:
        tweet['reply_count'] = metric['reply_count']
    if 'like_count' in metric:
        tweet['like_count'] = metric['like_count']
    if 'quote_count' in metric:
        tweet['quote_count'] = metric['quote_count']
    if 'impression_count' in metric:
        tweet['impression_count'] = metric['impression_count']
        
    return tweet
   
    
def set_tweet_values(values):
    '''
    Set the values of tweet object
    
    :return dictionary
    '''
    
    tweet = get_empty_tweet_dict()
    
    if 'conversation_id' in values:
        tweet['conversation_id'] = values['conversation_id']
        
    tweet['text'] = values['text']
    tweet['lang'] = values['lang']
    tweet['possibly_sensitive'] = values['possibly_sensitive']
    tweet['reply_settings'] = values['reply_settings']
    tweet['created_at'] = values['created_at']
    
    if 'edit_history_tweet_ids' in values:
        tweet['edit_history_tweet_ids'] = values['edit_history_tweet_ids']
        
    tweet['tweetid'] = values['id']
    tweet['author_id'] = values['author_id']
    
    if 'in_reply_to_user_id' in values:
        tweet['in_reply_to_user_id'] = values['in_reply_to_user_id']
    if 'referenced_tweets' in values:
        tweet['referenced_tweets'] = []
        for twt in values['referenced_tweets']:
            tweet['referenced_tweets'].append([twt['type'], twt['id']])
    
    tweet = add_entity_values(tweet, values)
    tweet = add_public_metric(tweet, values)
    
    if 'context_annotations' in values:
        tweet['context_annotations'] = values['context_annotations']
            
    return tweet

def set_values_for_tweet_with_error(values):
    '''
    Sets values for tweets not found
    :param values: return object of twitter API
    
    :return dictionary
    '''

    if values['resource_type'] != 'tweet':
        return None
    
    if 'referenced_tweets' in values['detail']:
        return None
    
    tweet = get_empty_tweet_dict()
    tweet['tweetid']= values['value']
    tweet['text'] = values['detail']
    tweet['author_id'] = None
    tweet['created_at'] = values['title']
    
    return tweet
    

def parse_tweets(tweet_file, output_file=None):
    '''
    Parse the tweets
    :param tweet_file: Raw tweet file
    :param output_file: File to be saved. If none no file is saved
    
    :return Dataframe
    '''
    all_tweets = []
    total = 0
    with open(tweet_file, 'r') as json_file:
        for row in json_file:
            one_row = json.loads(row)
            
            if 'errors' in one_row:
                for values in one_row['errors']:
                    tweet = set_values_for_tweet_with_error(values)
                    
                    if tweet != None:
                        all_tweets.append(tweet)
            
            if 'data' not in one_row:
                continue
            for tweet in one_row['data']:
                tweet_set = set_tweet_values(tweet)
                
                all_tweets.append(tweet_set)
                
    df = pd.DataFrame.from_records(data=all_tweets)
    
    if output_file is not None:
        df.to_pickle(output_file)

    return df          
                     
            
    
def extract_hashtags(df, 
                     column='tweet_text',
                     filter=False
                    ) -> pd.DataFrame:
    '''
    Extracts the hashtags from tweets
    
    :param df: Tweet Dataframe
    
    :return pandas Dataframe
    '''
    df[f'hashtags_{column}'] = df[column].apply(
        lambda x: list(set(re.findall(r'\B\#(\w+)', x))))
    
    if filter == True:
    
        return df.loc[df[f'hashtags_{column}'].map(len) != 0]
    return df



def get_user_info(path, save_path=None):
    '''
    Gets all user information from tweet json
    :param path: path to json files
    
    :return Dataframe
    '''
    
    all_profiles = []
    for file in glob.glob(path):
        print(file)
        with open(file, 'r') as json_file:
            for row in json_file:
                one_row = json.loads(row)
                if 'errors' in one_row:
                    for single_profile in one_row['errors']:
                        profile = set_values_for_profile_error(single_profile)
                    if profile != None:
                        all_profiles.append(profile)
                    
                if 'users' in one_row['includes']:
                    for user_row in one_row['includes']['users']:
                        profile = set_values_in_profile_dict(
                            user_row)
                        
                        if profile != None:
                            all_profiles.append(profile)
                
    df = pd.DataFrame.from_records(data=all_profiles)
    
    return df


def non_language():
    '''
    Get non lanaguage code in tweets
    '''
    return ['qam', #tweet with mention only
            'qct', #tweet with cashtags only
            'qht', #tweet with hashtags only
            'qme', #tweet with media links only
            'qst', #tweet with very short text
            'zxx', #absense of language code
            'und', #undetermined
                    # 'en'
            ]


def remove_non_language(df,
                        language_column='tweet_language'):
    '''
    Remove non language tweets
    :param df: DataFrame
    :param language_column: column that has language 
    information
    
    :return DataFrame
    '''
    non_lang = non_language()
    
    return df.loc[
        ~df[language_column].isin(non_lang)]