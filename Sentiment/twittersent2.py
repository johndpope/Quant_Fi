#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "970279222261542919-Xwk23YVIipDa2YhQtZxc6Fkh4wCEVAR"
access_token_secret = "L0ojWUHv0bJkYVyGSAKwF3rk9Gs8nQ4QWFlevsADGwcvK"
consumer_key = "sM5m6ocKaQZcvdeb1TDzbPJ4d"
consumer_secret = "tfzlkDVfMoqtDIphuATcti8KJtZyYG1yh30JGHqWBLPiejqCMT"

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(track=['python', 'javascript', 'ruby'])
