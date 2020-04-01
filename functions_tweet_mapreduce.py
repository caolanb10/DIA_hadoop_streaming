import datetime
import tweepy
import csv
import json
import pandas as pd
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import Twitter_API_Module as twt
from months import Month


# # # # # # # # # # # # #
# Extract:
# # # # # # # # # # # # #
def get_tweets_list(list_of_twitter_accs, num_pages):
    tweet_list = []
    for i in list_of_twitter_accs:
        tweet_list.append(twt.TwitterClientClass(twit_user=i).get_timeline_pages(num_pages))

    all = []
    for i in tweet_list:
        for ii in i:
            for iii in ii:
                all.append(iii)
    return all

def tweets_to_df(all):
    df_all_tweets = pd.DataFrame()
    df_all_tweets['TWEET_ID'] = [i.id for i in all]
    df_all_tweets['DATE_TIME'] = [str(i.created_at)[0:10] for i in all]
    df_all_tweets['TWITTER_ACC'] = [i.user.name for i in all]
    df_all_tweets['STR_ID'] = [i.id_str for i in all]
    df_all_tweets['FULL_TEXT'] = [i.full_text for i in all]
    df_all_tweets['HASHTAGS'] = [i.entities['hashtags'] for i in all]
    df_all_tweets['SOURCE'] = [i.source for i in all]
    df_all_tweets['FAV_COUNT'] = [i.favorite_count for i in all]
    df_all_tweets['RT_COUNT'] = [i.retweet_count for i in all]
    df_all_tweets['FOLLOWERS'] = [i.user.followers_count for i in all]
    df_all_tweets['TWEET_COUNT'] = [i.author.statuses_count for i in all]
    df_all_tweets['REPLY_TO_USER_ID'] = [i.in_reply_to_user_id for i in all]
    df_all_tweets['REPLY_TO_USER'] = [i.in_reply_to_screen_name for i in all]
    df_all_tweets['LEN_TWEET'] = [len(i) for i in df_all_tweets['FULL_TEXT']]
    df_all_tweets.sort_values(by='DATE_TIME', ascending=0)
    return df_all_tweets

def get_trump_json_data(local_filepath):
    with open(local_filepath, encoding="utf8") as json_trump_tweets:
        data = json.load(json_trump_tweets)
        return data

def get_json_data_to_df(data):
    df = pd.DataFrame()
    data_cols = [x for x in data[0]]
    df['SOURCE'] = [x.get('source') for x in data]
    df['TEXT'] = [x.get('text') for x in data]
    df['DATE']  = [x.get('created_at') for x in data]
    df['RETWEET_COUNT'] = [x.get('retweet_count') for x in data]
    df['FAVOURITE_COUNT'] = [x.get('favorite_count') for x in data]
    df['IS_RETWEETED'] = [x.get('is_retweet') for x in data]
    df['ID_STR'] = [x.get('id_str') for x in data]
    return df

# # # # # # # # # # # # #
# Explore Data:
# # # # # # # # # # # # #

def clean_text_words1(trump_df):
    # trump_df_cln = trump_df.drop(
    #     columns=['source', 'created_at', 'retweet_count', 'favorite_count', 'is_retweet', 'id_str'], axis=1)
    trump_df_cln['PROCESSED_TEXT'] = trump_df_cln['FULL_TEXT'].map(lambda i: re.sub('[,\.!?"]', '', i))
    for i in trump_df_cln['PROCESSED_TEXT']:
        i.replace('"', '')
    #trump_df_cln = trump_df_cln.drop(columns=['text'])
    return (trump_df_cln)

def clean_text_words(trump_df_all):
    trump_df_cln = pd.DataFrame()
    trump_df_cln['SOURCE'] = trump_df_all['SOURCE']
    trump_df_cln['PROCESSED_TEXT'] = trump_df_all['FULL_TEXT'].map(lambda i: re.sub('[,\.!?""@“”]', '', i))
    trump_df_cln['DATE'] = tweet_date_format(trump_df_all)
    trump_df_cln['RETWEET_COUNT'] = trump_df_all['RETWEET_COUNT']
    trump_df_cln['FAVOURITE_COUNT'] = trump_df_all['FAVOURITE_COUNT']
    trump_df_cln['IS_RETWEETED'] = trump_df_all['IS_RETWEETED']
    trump_df_cln['ID_STR'] = trump_df_all['ID_STR']
    return (trump_df_cln)

def clean_text_words3(trump_df_all):
    trump_df_all['PROCESSED_TEXT'] = trump_df_all['FULL_TEXT'].map(lambda i: re.sub('[,\.!?""@“”]', '', i))
    #trump_df_all['DATE'] = tweet_date_format(trump_df_all)
    return (trump_df_all)

def tweet_date_format(trump_df_all):
    yr = []
    day = []
    mnt = []
    for i in trump_df_all['DATE']:
        yr.append(i[26:30])
        day.append(i[8:10])
        month_text = i[4:7]
        mnt.append(Month[month_text])
    df_dts = pd.DataFrame({"Year": yr, "Month": mnt, "Day": day})
    new_dt = []
    for i in range(0,len(yr)):
        new_dt.append("-".join([df_dts['Year'][i], df_dts['Month'][i], df_dts['Day'][i]]))
    return new_dt

def get_wordcloud(trump_df_cln, filepath):
    make_string = ','.join(list(trump_df_cln['PROCESSED_TEXT'].values))
    word_cloud_obj = WordCloud(background_color="white", width=550, height=550, max_words=100, contour_width=2, contour_color='steelblue')
    word_cloud_obj.generate(make_string)
    word_cloud_obj.to_file(filepath)
    return word_cloud_obj.generate(make_string)

def get_top_words(trump_df_clean):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(trump_df_clean['PROCESSED_TEXT'])
    single_words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(single_words))
    for i in count_data:
        total_counts += i.toarray()[0]
    count_dict = (zip(single_words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:50] # removing 'https',  'tco', 'rt',
    single_words = [i[0] for i in count_dict]
    counts = [i[1] for i in count_dict]
    return [single_words, counts]

def get_sentiment_pa(trump_df_clean):
    sentiment = []
    for i in trump_df_clean['PROCESSED_TEXT']:
        blob = TextBlob(i)
        sentiment.append(blob.sentiment.polarity)
    trump_df_clean['SENTIMENT_PA'] = sentiment
    return trump_df_clean

def get_sentiment_nbayes(trump_df_clean):
    sentiment = []
    '''extremely slow'''
    for i in trump_df_clean['PROCESSED_TEXT']:
        blob = TextBlob(i, analyzer = NaiveBayesAnalyzer())
        sentiment.append(blob.sentiment.classification)
    trump_df_clean['SENTIMENT_NB'] = sentiment
    return trump_df_clean
