#!/usr/bin/python

import pandas as pd
import sys

df_all = pd.read_csv('/home/tiernan/Documents/map_reduce/twitter_mass_media_data.csv')

df = pd.DataFrame()
df['DATE'] = df_all['DATE_TIME']
df['TEXT'] = df_all['FULL_TEXT']

df.to_csv('/home/tiernan/PycharmProjects/untitled/twitter_2cols.csv')


