""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Performs Multilingual Sentiment Analysis using XLM-Roberta-Large-XNLI-ANLI
####################################################################################################################"""
import re
import time
from datetime import datetime, timedelta

import pandas as pd
from pymongo import MongoClient

import utils

start_time = time.time()

from transformers import pipeline
model = "path/to/model"
classifier = pipeline("zero-shot-classification",
                      model, device=0)


# extract text field of twitter object
def get_text(tweet):
    if "extended_tweet" in tweet.keys() and "full_text" in tweet["extended_tweet"].keys():
        # case of extended tweet object
        return tweet['extended_tweet']['full_text']
    elif "retweeted_status" in tweet.keys() and "extended_tweet" in tweet["retweeted_status"].keys() and "full_text" in \
            tweet["retweeted_status"]["extended_tweet"].keys():
        # case of retweet object with extednded status
        return tweet['retweeted_status']['extended_tweet']['full_text']
    elif "retweeted_status" in tweet.keys() and "full_text" in tweet["retweeted_status"].keys():
        # case of retweetedd object with full_text
        return tweet['retweeted_status']['full_text']
    elif "retweeted_status" in tweet.keys():
        tweet_text = tweet["full_text"] if "full_text" in tweet else tweet["text"]
        tweet_text = utils.merge_tw_rt(tweet_text,
                                       tweet["retweeted_status"]["full_text"] if "full_text" in
                                                                                 tweet["retweeted_status"] else
                                       tweet["retweeted_status"]["text"])
        return tweet_text
    elif "full_text" in tweet.keys():
        # tweet object with full_text
        return tweet['full_text']
    elif "text" in tweet.keys():
        # case of simple text field in tweet object
        return tweet['text']
    return None


def text_cleanup(tweet_text):
    clean_tweet_text = re.sub(r"http\S+", "", tweet_text)
    clean_tweet_text = re.sub('[@#]', '', clean_tweet_text)
    return clean_tweet_text


def getaveragebyday(df):
    df = df.groupby([pd.Grouper(key='day', freq='D'), 'Sentiment']) \
        .size().unstack('Sentiment')
    df['sum'] = df.sum(axis=1)
    df['PosUkraine'] = df['PosUkraine'] / df['sum'] * 100
    df['PosRussia'] = df['PosRussia'] / df['sum'] * 100
    df['NegUkraine'] = df['NegUkraine'] / df['sum'] * 100
    df['NegRussia'] = df['NegRussia'] / df['sum'] * 100
    df = df.drop('sum', 1)
    return df


def xlmR_classifier(db):
    df_sentiment = pd.DataFrame(columns=['day', 'Sentiment'])
    sentiment_labels = ['positive for ukraine', 'negative for ukraine', "neutral for ukraine",
                        'positive for zelensky', 'negative for zelensky', "neutral for zelensky",
                        'positive for putin', 'negative for putin', "neutral for putin",
                        'positive for russia', 'negative for russia', "neutral for russia"]
    start_date = datetime(2022, 2, 24, 0, 0, 0)
    end_date = datetime(2022, 2, 25, 0, 0, 0)
    while end_date < datetime(2022, 3, 6, 0, 0, 0):
        end_day = start_date + timedelta(days=1)
        hour_start = start_date
        hour_end = hour_start + timedelta(hours=4)
        print("working on day {} - {}".format(start_date, end_day))
        while hour_end <= end_day:
            tweetText_list = []
            tweetsDate_list = []
            print("\t hours {} {}".format(hour_start, hour_end))
            for tweet in db.Tweets.find({"created_at": {"$gte": hour_start, "$lt": hour_end}}):
                tweet_text = get_text(tweet)
                clean_tweet_text = text_cleanup(tweet_text)
                if len(clean_tweet_text) > 3:
                    tweetText_list.append(clean_tweet_text)
                    tweetsDate_list.append(tweet["created_at"])

            sentiment = classifier(tweetText_list, sentiment_labels, batch_size=4, gradient_accumulation_steps=4,
                                   gradient_checkpointing=True, fp16=True, optim="adafactor")
            # print(sentiment)
            print("--- Time spend: {:.4f} in hours ---".format((time.time() - start_time) / 3600.0))
            for i, j in zip(sentiment, tweetsDate_list):
                if i['labels'][0] == "positive for ukraine" or i['labels'][0] == "positive for zelensky":
                    new_row = {'day': j, 'Sentiment': "PosUkraine", "score": i['scores'][0]}
                    df_sentiment = df_sentiment.append(new_row, ignore_index=True)
                elif i['labels'][0] == "positive for russia" or i['labels'][0] == "positive for putin":
                    new_row = {'day': j, 'Sentiment': "PosRussia", "score": i['scores'][0]}
                    df_sentiment = df_sentiment.append(new_row, ignore_index=True)
                elif i['labels'][0] == "negative for russia" or i['labels'][0] == "negative for putin":
                    new_row = {'day': j, 'Sentiment': "NegRussia", "score": i['scores'][0]}
                    df_sentiment = df_sentiment.append(new_row, ignore_index=True)
                elif i['labels'][0] == "negative for ukraine" or i['labels'][0] == "negative for zelensky":
                    new_row = {'day': j, 'Sentiment': "NegUkraine", "score": i['scores'][0]}
                    df_sentiment = df_sentiment.append(new_row, ignore_index=True)
            hour_start += timedelta(hours=4)
            hour_end += timedelta(hours=4)
            del (tweetText_list)
            del (tweetsDate_list)
        df_sentiment["day"] = pd.to_datetime(df_sentiment["day"])
        df_average = getaveragebyday(df_sentiment)
        df_average.to_csv("sentiment_by_day_{}_{}.csv".format(start_date, end_day).replace(" ", "_"), sep="\t",
                          header=True)
        start_date += timedelta(days=1)  # move to next day
        del (df_sentiment)
        del (df_average)


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27017')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


if __name__ == '__main__':
    # connect to mongo DB
    client, db = connect_to_db()
    xlmR_classifier(db)
    print("--- %s seconds ---" % (time.time() - start_time))
