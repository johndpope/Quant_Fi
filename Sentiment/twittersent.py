#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:16:29 2018

@author: nicob
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3
from unidecode import unidecode
import time
from textblob import TextBlob

#consumer key, consumer secret, access token, access secret.
ckey="sM5m6ocKaQZcvdeb1TDzbPJ4d"
csecret="tfzlkDVfMoqtDIphuATcti8KJtZyYG1yh30JGHqWBLPiejqCMT"
atoken="970279222261542919-Xwk23YVIipDa2YhQtZxc6Fkh4wCEVAR"
asecret="L0ojWUHv0bJkYVyGSAKwF3rk9Gs8nQ4QWFlevsADGwcvK"

conn = sqlite3.connect('/Users/nicob/Desktop/twitter_all.db')
c = conn.cursor()

def create_table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))
create_table()



class listener(StreamListener):

    def on_data(self, data):
        try:
            data = json.loads(data)
            tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            
            analysis = TextBlob(tweet)
            sentiment = analysis.sentiment.polarity
            
            print(time_ms, tweet, sentiment)
            c.execute("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)",
                  (time_ms, tweet, sentiment))
            conn.commit()

        except KeyError as e:
            print(str(e))
        return(True)

    def on_error(self, status):
        print(status)


while True:

    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener())
        twitterStream.filter(track=["TPMP", "#TPMP"])
    except Exception as e:
        print(str(e))
        time.sleep(5)