import re
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punction(text):
    symbols = str.maketrans('', '', string.punctuation)
    
    return text.translate(symbols)

def remove_repeated_letters(text):
    letters = re.compile(r'([A-za-z])\1{2,}')
    
    return letters.sub(r'\1', text)

def emoji_in_symbol(text):
    emoji = re.compile(r'[8:=;<][\'\-]?[\[\])dDpXx(\\/3<>]?')
    
    return emoji.sub(r'', text)

def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in (stopwords)])
    
    return text


#useful for getting just english tweets
def remove_not_ASCII(text):
    text = ''.join([word for word in text if word in string.printable])

    return text

def remove_mentions(text):
    rm = re.compile(r'@\S+')
    
    return rm.sub(r'', text)

def remove_hashtags(text):
    rm = re.compile(r'#\S+')
    
    return rm.sub(r'', text)

def remove_html_tags(text):
    tags = re.compile(r'<.*?>')
    
    return tags.sub(r'', text)

def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    
    return num.sub(r'', text)

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.S+')
    
    return url.sub(r'', text)


def remove_rt(text):
    rt = re.compile(r'\b^rt\b')
    
    return rt.sub(r'', text)


def clean_tweet(text):
    text = remove_URL(text)
    text = remove_html_tags(text)
    text = remove_mentions(text)
    
    text = remove_number(text)
    
    text = remove_emoji(text)
    text = emoji_in_symbol(text)
    
    text = remove_punction(text)
    text = remove_repeated_letters(text)
    
    text = remove_stopwords(text)
    
    
    
    if len(text.strip()) == 0:
        text = 0
    
    return text.lower()