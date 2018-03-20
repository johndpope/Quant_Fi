#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:27:57 2018

@author: nicob
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('twitter_all.db')
c = conn.cursor()

df = pd.read_sql("SELECT * FROM sentiment ORDER BY unix DESC LIMIT 1000", conn)
df.sort_values('unix', inplace=True)

df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
df.dropna(inplace=True)
print(df.tail())