#!/usr/bin/python

# DES: Reducer script to find average sentiment for any given day.
# BY:  Tiernan Barry, x19141840 - NCI.

# 1. Libraries:
from operator import itemgetter
import sys

# 2. reduce key,values by date and find average score per date:
last_date_key = None
aggregate_sentiment = 0
count_per_date = 0

for sentiment in sys.stdin:
    sentiment = sentiment.strip()  # if whitespace - removes
    this_date_key, sentiment_value = sentiment.split()  # splits mapper by tab escaped
    sentiment_value = float(sentiment_value)

    if last_date_key == this_date_key:
        count_per_date += 1
        aggregate_sentiment += sentiment_value
    else:
        if last_date_key:
            print(('%s\t%s\t%s') % (last_date_key, aggregate_sentiment / count_per_date, count_per_date))
        aggregate_sentiment = sentiment_value
        last_date_key = this_date_key
        count_per_date = 1

# -- Output the least popular / min count sentiment sentiment
if last_date_key == this_date_key:
    print(('%s\t%s\t%s') % (last_date_key, aggregate_sentiment / count_per_date, count_per_date))

#########################################################

