""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Performs Topic Modelling
####################################################################################################################"""
import gc
import re
from datetime import datetime, timedelta
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from pymongo import MongoClient
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

import utils
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def text_cleaning(text):
    """
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    """
    text = re.sub("#\w+", "", text)
    text = re.sub(r'\brt\b', '', text)
    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r"[^a-zA-Z\d]", " ", text)  # Remove special Charecters
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces
    text = text.strip()  # remove spaces at the beginning and at the end of string

    return text


# takes a list and integer n as input and returns generator objects of n lengths from that list
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)

    result_list_tqdm = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'day'])
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm = pd.concat([result_list_tqdm, result], axis=0, ignore_index=True)

    return result_list_tqdm


def calculate_docs(start, end):
    client, db = connect_to_db()
    return [x['_id'] for x in db.Tweets.find({"created_at": {"$gte": start, "$lt": end}}, {'_id': 1})]


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


def calculate(chunk):
    df_military = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'day'])
    # define client inside function
    client, db = connect_to_db()
    # do the calculation with document collection.find_one(id)
    result = db.Tweets.find({"id": {'$in': chunk}},
                            {'id': 1, 'user': 1, 'text': 1, 'extended_tweet': 1, 'retweeted_status': 1,
                             'created_at': 1, 'lang': 1})
    tweetList = list(result)
    for tweet in tweetList:
        text = get_text(tweet)
        if text is not None:

            text = text_cleaning(text)
            if text is not None:
                new_row = {'user_id': tweet['user']['id'], 'tweet_id': tweet['id'], 'text': text,
                           'created_at': tweet['created_at']}
                df_temp = pd.DataFrame([new_row])
                df_military = pd.concat([df_military, df_temp], axis=0, ignore_index=True)
    return df_military


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27007')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


def bertTopics(df, start_date):
    stopw = stopwords.words()
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopw)

    topic_model = BERTopic(vectorizer_model=vectorizer_model, low_memory=True, calculate_probabilities=False,
                           verbose=True,
                           nr_topics='auto',
                           n_gram_range=(1, 2)
                           , min_topic_size=300)
    topics = topic_model.fit_transform(df["text"].to_list())
    fig = topic_model.visualize_topics(top_n_topics=20)
    fig.write_html('distance_map_{}.html'.format(start_date))
    fig = topic_model.visualize_hierarchy(top_n_topics=20)
    fig.write_html('topics_hierarchy_{}.html'.format(start_date))
    fig = topic_model.visualize_barchart(top_n_topics=20)
    fig.write_html('word_scores_{}.html'.format(start_date))


def start():

    list_of_dates = [datetime(2022, 2, 25, 0, 0, 0),
                     datetime(2022, 6, 3, 0, 0, 0), datetime(2022, 3, 4, 0, 0, 0)]
    for start_date in list_of_dates:
        end_date = start_date + timedelta(days=1)
        df_military = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'day'])
        print("working on day {}".format(start_date))
        document_ids = calculate_docs(start_date, end_date)
        df_military = pd.concat(
            [df_military, run_imap_multiprocessing(calculate, list(chunks(document_ids, 1000)), 10)], axis=0,
            ignore_index=True)
        df_military.replace("", np.nan, inplace=True)
        df_military.dropna(subset=["text"], inplace=True)
        bertTopics(df_military, start_date)
        del df_military
        gc.collect()


if __name__ == '__main__':
    start()
