#!/usr/bin/python

# DES: Mapper script which reads in a list of tweets, and applies a sentiment score for each tweet.
#      Returns key value pair of sentiment score and the date of tweet.
# BY:  Tiernan Barry, x19141840 - NCI.

# 1. Libraries:
import sys
from textblob import TextBlob
import pandas as pd
import re

# 2. Transform sys.stdin from class '_io.textiowrapper' to dataframe for easier handling.
all = []
# for i in df['0']: # if testing in Pycharm
for i in sys.stdin:
    all.append(i)

# 3. Split data between Date and Text:
all = all[1:len(all)] # Remove heading
date = all[:int((len(all))/2)]
text = all[int((len(all))/2):]

# 4. Mapper: Clean data and generate the sentiment in key/value pair.
df = pd.DataFrame({"DATE": date, "TEXT": text})
dt_format = []
for i in df["DATE"]:
    date = str(i)
    date_frmt = date[0:10]
    dt_format.append(date_frmt)

# Key = date, value = sentiment.
for i in range(0,len(df["TEXT"])):
    blob = TextBlob(df["TEXT"][i])
    sent = blob.sentiment.polarity # sentiment = key
    value = sent # if going with -1,0,1 remove this line
    key = dt_format[i]
    print(('%s\t%s') % (key, value))
