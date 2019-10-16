import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from google.cloud import language_v1
from google.cloud.language_v1 import enums
from google.cloud import storage

#Initialize Storage Client and download tweets csv file
storage_client = storage.Client('interview-flighthub').from_service_account_json('/home/lwalls/documents/apikey.json')
bucket = storage_client.get_bucket('fh-interview-speech-audio')
blob = bucket.blob('flighthub_tweets_raw.csv')
blob.download_to_filename('/home/lwalls/documents/natlang/flighthub_tweets_raw.csv')

# Place Tweets in Df and clean
#df = pd.read_csv('gs://fh-interview-speech-audio/flighthub_tweets_raw.csv')
df = pd.read_csv('flighthub_tweets_raw.csv')
df = df.drop(['replies', 'retweets', 'favorites', 'unix_timestamp', 'url', '__url'], axis=1)

## clean dates
twitter_dates = df['date']
twitter_dates_converted = []
for d in twitter_dates:
    if '2018' in d:
        d = datetime.strptime(d, '%d %b %Y').date()
        d.strftime('%Y-%m-%d')
        twitter_dates_converted.append(d)
    else:
        d = datetime.strptime(d, '%b %d').date()
        d.strftime('%Y-%m-%d')
        d = d.replace(year=datetime.today().year)
        twitter_dates_converted.append(d)

df['date'] = twitter_dates_converted

## sort by newest to oldest
df = df.sort_values('date', ascending=False)


# Sentiment Analysis
def language_analysis(text_content):
    client = language_v1.LanguageServiceClient.from_service_account_json('/home/lwalls/documents/apikey.json')
    type_ = enums.Document.Type.PLAIN_TEXT
    language = 'en'
    document = {'content': text_content, 'type': type_, 'language': language}

    encoding_type = enums.EncodingType.UTF8
    response_sentiment = client.analyze_sentiment(document, encoding_type=encoding_type)
    #sent_analysis = document.analyze_sentiment()
   # print(dir(response_sentiment))
    sentiment = response_sentiment.document_sentiment
    #ent_analysis = document.analyze_entities()

    response_ent = client.analyze_entities(document, encoding_type=encoding_type)
    entities = response_ent.entities
    return sentiment, entities

tweets = df['content']
sentiment_score = []
sentiment_magnitude = []
df_entities = pd.DataFrame()

for tweet in tweets:
    # Sentiment
    sentiment, entities = language_analysis(tweet)
    sentiment_score.append(sentiment.score)
    sentiment_magnitude.append(sentiment.magnitude)
    
    # Entities
    entity_name = []
    entity_type = []
    entity_salience = []
    df_ent_tmp = pd.DataFrame()
    for e in entities:
        entity_name.append(e.name)
        entity_type.append(enums.Entity.Type(e.type).name)
        entity_salience.append(e.salience)
    df_ent_tmp['name'] = entity_name
    df_ent_tmp['type'] = entity_type
    df_ent_tmp['salience'] = entity_salience
    df_ent_tmp['tweet'] = tweet
    
    df_entities = pd.concat([df_entities, df_ent_tmp], ignore_index=True)
    del df_ent_tmp

df['sentiment_score'] = sentiment_score
df['sentiment_magnitude'] = sentiment_magnitude    
print(df.head(10))
print(df_entities.head(10))

df.to_csv('flighthub_tweets_sentiment.csv')
df_entities.to_csv('flighthub_tweets_entities.csv')
