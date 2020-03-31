# pip install textblob
import twitter_module as twt
import numpy as np
import pandas as pd
import re
import missingno as msno
import functions as fns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

###########################################
# Extract:
###########################################

irish_tweets_list = fns.get_irish_tweets_list("Ireland", 100)

df_irish_tweets = fns.tweets_to_df(irish_tweets_list)

df_irish_tweets.to_csv(r"C:\Users\btier\PycharmProjects\Data_Intensive_Architecture\irish_public_tweets.csv", index=False)

###########################################
# Transform:
###########################################

# Check for NAs
#msno.matrix(df_irish_tweets)

# Make new column for processed name - to be sent to mapper:
df_irish_tweets['PROCESSED_TEXT'] = df_irish_tweets['FULL_TEXT'].map(lambda i: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", '', i))

# Check for formatting:
#irish_word_cloud = fns.get_wordcloud(df_irish_tweets, r"C:\Users\btier\Documents\irish_word_cloud.png")

standard_in = pd.DataFrame({"DATE": df_irish_tweets['DATE_TIME'], "TEXT": df_irish_tweets['PROCESSED_TEXT']})

print(standard_in)






