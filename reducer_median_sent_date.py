#!/usr/bin/python

# DES: Reducer script to find average sentiment for any given day.
# BY:  Tiernan Barry, x19141840 - NCI.

# 1. Libraries:
from operator import itemgetter
import sys
from sortedcontainers import SortedList

# 2. reduce key,values by date and find median score per date:
last_date_key = None
sent_list_sort = SortedList()
count_per_date = 0

for sentiment in sys.stdin:
    sentiment = sentiment.strip()  # if whitespace - removes
    this_date_key, sentiment_value = sentiment.split()  # splits mapper by tab escaped
    sentiment_value = float(sentiment_value)

    if last_date_key == this_date_key:
        count_per_date += 1
        sent_list_sort.add(sentiment_value) #1
    else:
        if last_date_key:
            print(('%s\t%s\t%s') % (last_date_key, sent_list_sort[int(len(sent_list_sort)/2)], count_per_date)) #2
        #aggregate_sentiment = sentiment_value #3
        last_date_key = this_date_key
        count_per_date = 1

# -- Output the least popular / min count sentiment sentiment
if last_date_key == this_date_key:
    print(('%s\t%s\t%s') % (last_date_key, sent_list_sort[int(len(sent_list_sort)/2)], count_per_date)) #4
