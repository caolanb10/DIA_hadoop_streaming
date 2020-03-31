# Des: Source script for extracting 2 datasets for Data Intensive Architecture - Map Reduce project.
#      1. Twitter data from global media twitter pages.
#      2. Read in Trump data
# By: Tiernan Barry - x19141840 - NCI.

# Libaries, imported files and installations (if required):
# pip install textblob
import functions_tweet_mapreduce as fns # Set of functions defined for this project
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

######################################################################################
# Extract: 1. Twitter data from global media twitter pages.
######################################################################################

twitter_pgs = ["CNN", "BBCWorld", "BBCBreaking", "BBCNews", "ABC", "itvnews", "Independent",
               "RTENewsNow", "Independent_ie", "guardian", "guardiannews", "rtenews", "thejournal_ie",
               "wef", "IMFNews", "WHO", "euronews", "MailOnline", "TheSun", "Daily_Express", "DailyMirror",
               "standardnews", "LBC", "itvnews", "thetimes", "IrishTimes", "ANI", "XHNews", "TIME", "OANN",
               "BreitbartNews", "Channel4News", "BuzzFeedNews", "NewstalkFM", "NBCNewsBusiness", "CNBCnow",
               "markets", "YahooFinance", "MarketWatch", "Forbes", "businessinsider", "thehill", "CNNPolitics",
               "NPR", "AP", "USATODAY", "NYDailyNews", "nypost",
               "CBSNews", "MSNBC", "nytimes", "FT", "business", "cnni", "RT_com", "AJEnglish", "CBS", "NewsHour",
               "NPR", "BreakingNews", "cnnbrk", "WSJ", "Reuters", "SkyNews", "CBCAlerts"]

tweets_list = fns.get_tweets_list(twitter_pgs, 1) # change to 90 for full dataset +

df_all_tweets = fns.tweets_to_df(tweets_list)

df_all_tweets = df_all_tweets.sort_values(by='DATE_TIME', ascending=0)

# Test:
df_all_tweets.to_csv("/home/tiernan/PycharmProjects/DIA/twitter_mass_media_data_TEST.csv", index= False)

# Real:
# df_all_tweets.to_csv("/home/tiernan/PycharmProjects/DIA/twitter_mass_media_data.csv", index= False)


