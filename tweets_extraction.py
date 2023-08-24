import tweepy
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt') #punkt tokenizer model
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
import emoji
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


# authentication object using the 4 keys
def twitter_setup(consumer_key, consumer_secret, access_key, access_secret):
    
    # Authentication and access using keys
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key,access_secret)
    api= tweepy.API(auth,wait_on_rate_limit=True)
    
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during authentication")
    return api

# tweet extraction
def keyword_tweets(api,keyword,number_of_tweets):
    new_keyword=keyword+' -filter:retweets'
    
    # initiate DataFrame
    df = pd.DataFrame()
    tweets={}

    # use search function instead of user timeline
    for status in tweepy.Cursor(api.search_tweets, q=new_keyword, lang="en", tweet_mode='extended', result_type='mixed').items(number_of_tweets):
        tweets['Tweets'] = status.full_text
        tweets['Tweets_ID'] = status.id.astype(str)
        tweets['Date'] = status.created_at
        tweets['Source'] = status.source
        tweets['Likes_no'] = status.favorite_count
        tweets['Retweets_no'] = status.retweet_count
        tweets['Hashtag'] = status.entities['hashtags']
        tweets['Location'] = status.user.location
        tweets['Place'] = status.place
        tweets['UID'] = status.user.id
        tweets['Username'] = status.user.screen_name
        tweets['DisplayName'] = status.user.name
        tweets['Verified'] = status.user.verified
        
        # append to the dataframe
        df = df.append(tweets, ignore_index=True)
    return df 

# tweet cleaning
def remove_duplicates(df):
    print(f'Before dropping duplicates: {df.shape}')
    df = df.drop_duplicates(subset=('Tweets'))
    print(f'After dropping duplicates: {df.shape}')
    return df


def cleaning_location_tweets(df):
    location_to_country = {
        'United States': r'(?:^|\W)(us|usa|united states|america|ny|new york|nyc|ohio|oh|nashville|tn|midwest|dc|md|toronto|nj|newark|hawaii|nc|wi|arizona|ak|pa|ga|texas|tx|california|alaska|lauderdale|illinois|carolina|in|indiana|manhattan|az|fl|florida|puerto rico|maryland|oklahoma|ms|mississippi|mi|michigan|idaho|boise|atlanta|oklahoma|sc|genesee|maryland|tennessee|ky|kentucky|brooklyn)(?:$|\W)',
        'United Kingdom': r'(?:^|\W)(uk|united kingdom|england|dunstable|london|maidstone|liverpool|essex|wales|scotland|birmingham|yorkshire)(?:$|\W)',
        'Ireland': r'(?:^|\W)(ireland|dublin|portstewart)(?:$|\W)',
        'Australia': r'(?:^|\W)(australia|new south wales|melbourne|wurundjeri|sydney|aus)(?:$|\W)',
        'Malaysia': r'(?:^|\W)(malaysia|my|mas|petaling jaya|pj|negeri|setiap hari|selangor|johor|kl|kuala lumpur|cyberjaya|kelantan)(?:$|\W)',
        'Singapore': r'(?:^|\W)singapore(?:$|\W)',
        'Philippines': r'(?:^|\W)(philippines|ph|mandaluyong|diliman|quezon)(?:$|\W)',    
        'Ghana': r'(?:^|\W)(ghana|accra)(?:$|\W)',
        'Nigeria': r'(?:^|\W)nigeria(?:$|\W)',
        'Other African Countries': r'(?:^|\W)(africa|za)(?:$|\W)',
        'Zimbabwe': r'(?:^|\W)zimbabwe(?:$|\W)',
        'Kenya': r'(?:^|\W)kenya(?:$|\W)',
        'Algeria': r'(?:^|\W)(algeria|annaba)(?:$|\W)',           
        'Uganda': r'(?:^|\W)uganda(?:$|\W)',
        'Kuwait': r'(?:^|\W)kuwait(?:$|\W)',
        'Japan': r'(?:^|\W)(tokyo|japan)(?:$|\W)',
        'Germany': r'(?:^|\W)(germany|schleswig-holstein)(?:$|\W)',
        'South Korea': r'(?:^|\W)(south korea|daegu|seoul)(?:$|\W)',
        'North Korea': r'(?:^|\W)north korea(?:$|\W)',
        'Egypt': r'(?:^|\W)(egypt|alexandria)(?:$|\W)',
        'Saudi Arabia': r'(?:^|\W)saudi arabia(?:$|\W)',
        'France': r'(?:^|\W)(france|paris)(?:$|\W)',
        'Sri Lanka': r'(?:^|\W)(sri lanka|lk)(?:$|\W)',
        'Thailand': r'(?:^|\W)(thailand|bangkok)(?:$|\W)',
        'Belgium': r'(?:^|\W)belgium(?:$|\W)',
        'Hungary': r'(?:^|\W)(hungary|budapest)(?:$|\W)',
        'Netherlands': r'(?:^|\W)(netherlands|den haag)(?:$|\W)',
        'Jamaica': r'(?:^|\W)jamaica(?:$|\W)',
        'Spain': r'(?:^|\W)(pain|spain)(?:$|\W)',  
        'New Zealand': r'(?:^|\W)(new zealand|nz|auckland)(?:$|\W)',            
        'Canada': r'(?:^|\W)(canada|toronto|winnipeg|alberta)(?:$|\W)',            
        'India': r'(?:^|\W)(india|mumbai)(?:$|\W)',   
        'Cyprus': r'(?:^|\W)cyprus(?:$|\W)'
    }

    def extract_country(location):
        for country, pattern in location_to_country.items():
            if re.search(pattern, location.lower()):
                return country
        return None
    
    df['Place'] = df['Location'].apply(extract_country)
    return df


def cleaning_text_tweets(df):
    
    # lemmatisation
    def lemmatize_sentence(token):
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence=[]
        
        for word, tag in pos_tag(token):
            if tag.startswith('NN'):
                pos='n'
            elif tag.startswith('VB'):
                pos='v'
            else:
                pos='a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
        return lemmatized_sentence
    
    # remove noises from the tweets like links, mentions, numbers, punctuations, stopwords
    def remove_noise(tweet_tokens, stop_words):
        cleaned_tokens=[]
        for token in tweet_tokens:
            # remove http/https links
            token = re.sub('http([!-~]+)?','',token) 

            # remove t.co and anything behind it
            token = re.sub('//t.co/[A-Za-z0-9]+','',token) 

            # remove all @ mentions
            token = re.sub('(@[A-Za-z0-9_]+)','',token)

             # remove all numbers
            token = re.sub('[0-9]','',token)

            # remove any non normal UT8 characters
            token = re.sub('[^ -~]','',token) 
            
            # remove emojis (just in case)
            token = re.sub(emoji.get_emoji_regexp(), "", token)

            #remove \n and \r and lowercase
            token = token.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() 

            # remove non utf8+ascii characters (triple confirm)
            token = re.sub('[^\x00-\x7f]','', token) 

            # turn double spaces to 1
            token = re.sub("\s\s+" , " ", token)

            # if condition to keep words with more than 3 char count and is not punctuations and in stopwords list
            if (len(token)>3) and (token not in string.punctuation) and (token.lower() not in stop_words):
                cleaned_tokens.append(token.lower())
        return cleaned_tokens
    
    # initiate stopwords
    stop_words=stopwords.words('english')
    stop_words.extend(['video','account','social media', 'social medium','sosmed','people','also',
                    '/like','/comment','/subscribe','/tiktok','/youtube','twitter','instagram','tiktok',
                    'do like','do follow', 'please like','please follow','please',
                    'need', 'followers', 'likes', 'views', 'shares', 'subscribers', 'follow','follows'
                    'Instagram', 'twitter', 'shopee', 'youtube', 'facebook',
                    'you tube','make','n e e d','check it out', 'check', 'check out','checks'
                    'know','go','watch','videos','going','say','saying','said','says'])
    
    # perform cleaning
    tweets_token=df['Tweets'].apply(word_tokenize).tolist()
    
    cleaned_tokens=[]
    for token in tweets_token:
        rm_noise =remove_noise(token, stop_words)
        lemma_tokens=lemmatize_sentence(rm_noise)
        cleaned_tokens.append(lemma_tokens)
    
    tweet_list = [tweet for tweet in cleaned_tokens if tweet!='[]']
    df['Tweets_Cleaned'] = tweet_list
    return df


if __name__ == '__main__':
    # replace with credentials
    consumer_key = "<replace-here>"
    consumer_secret = "<replace-here>"
    access_key = "<replace-here>"
    access_secret = "<replace-here>"
    
    # setup twitter api
    extractor = twitter_setup(consumer_key, consumer_secret, access_key, access_secret)
    
    # search keywords
    tiktok_alltweets=keyword_tweets(extractor,"TikTok -please -check until:2022-07-03 since:2022-07-02",2400)

    # clean tweets
    data = tiktok_alltweets.copy()
    data = cleaning_location_tweets(data)
    data = remove_duplicates(data)
    data = cleaning_text_tweets(data)

    # export as csv
    data.to_csv("TikTokUserTweets.csv")
