import numpy as np
import pandas as pd
import re
import missingno as msno
import functions_tweet_mapreduce as fns
import matplotlib.pyplot as plt
import warnings

##########################################################################
# Extract:
##########################################################################

# -- Read in Media tweets:
media_tweets = pd.read_csv("/home/tiernan/PycharmProjects/DIA/twitter_mass_media_data.csv")

##########################################################################
# Transform:
##########################################################################

# -- Deal with NA values: Back fill followed by forward fill
msno.matrix(media_tweets, figsize= (50,30))

# -- Make new column for processed name:
media_tweets['PROCESSED_TEXT'] = media_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))
#media_tweets[['DATE_TIME','PROCESSED_TEXT']].to_csv("/home/tiernan/PycharmProjects/DIA/media_processed_for_sent.csv", index = False)

list_all_data = []
for i in media_tweets['DATE_TIME']:
    list_all_data.append(i)
for i in media_tweets['PROCESSED_TEXT']:
    list_all_data.append(i)

df_1_col = pd.DataFrame(list_all_data)
df_1_col.to_csv("/home/tiernan/PycharmProjects/DIA/media_processed_for_sent.csv", index = False)

# -- Check for formatting using word cloud:
#word_cloud = fns.get_wordcloud(media_tweets, "/home/tiernan/PycharmProjects/DIA/twitter_media_wordcloud.png")


# -- Plot top words - removed stop words:
'''
top_words = fns.get_top_words(media_tweets)
df_top_words = pd.DataFrame({"WORD": top_words[0], "COUNT": top_words[1]})
plt.figure()
plt.bar(df_top_words["WORD"][0:10], df_top_words["COUNT"][0:10])
plt.xlabel('Words', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title("Top 10 Words", fontsize=20)'''

##########################################################################
# Analysis:
# Sentiment Analysis: Lexicon
#
##########################################################################

##########################################
# 1. Sentiment Analysis: Lexicon-based polarity
##########################################

# -- Lexicon-based sentiment (-1:1):
#trump_tweets = fns.get_sentiment_pa(trump_tweets)

# -- Get Average Sentiment for each day: Map/reduce?
#df_feature_1 = pd.DataFrame()