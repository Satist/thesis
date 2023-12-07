""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Performs Multilingual Sentiment Analysis using XLM-Roberta-Large-XNLI-ANLI
####################################################################################################################"""
import gc
import re
import time
from datetime import datetime, timedelta
from functools import partial
from itertools import chain
from multiprocessing import Pool, pool

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pymongo import MongoClient
from tqdm.auto import tqdm
import swifter
import utils
from torch.utils.data import Dataset

start_time = time.time()

from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="vicgalle/xlm-roberta-large-xnli-anli", device=0)


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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


def classifier_func(text):
    label = ['military']
    if text is not None:
        result = classifier(text, label, batch_size=4, gradient_accumulation_steps=16,
                            gradient_checkpointing=True, fp16=True, optim="adafactor")
        return result['scores'][0]
    return 0


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


def calculate_docs(hour_start, hour_end):
    client, db = connect_to_db()
    return db.Tweets.find({"created_at": {"$gte": hour_start, "$lt": hour_end}}).distinct('id')


def calculate(chunk):
    df_military = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'day'])
    # define client inside function
    client, db = connect_to_db()
    # do the calculation with document collection.find_one(id)
    result = db.Tweets.find({"id": {'$in': chunk}},
                            {'id': 1, 'user': 1, 'text': 1, 'extended_tweet': 1, 'retweeted_status': 1,
                             'created_at': 1})
    tweetList = list(result)
    for tweet in tweetList:
        text = get_text(tweet)
        if text is not None:
            text = text_cleaning(text)
            new_row = {'user_id': tweet['user']['id'], 'tweet_id': tweet['id'], 'text': text,
                       'created_at': tweet['created_at']}
            df_temp = pd.DataFrame([new_row])
            df_military = pd.concat([df_military, df_temp], axis=0, ignore_index=True)
    return df_military


def xlmR_classifier():
    label = ['military']
    start_date = datetime(2022, 7, 26, 0, 0, 0)
    end_date = datetime(2022, 7, 27, 0, 0, 0)
    while end_date < datetime(2022, 11, 30, 0, 0, 0):
        df_military = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'day'])
        end_day = start_date + timedelta(days=1)
        hour_start = start_date
        hour_end = hour_start + timedelta(hours=4)
        print("working on day {} - {}".format(start_date, end_day))
        while hour_end <= end_day:
            print("\t hours {} {}".format(hour_start, hour_end))
            document_ids = calculate_docs(hour_start, hour_end)
            df_military = pd.concat(
                [df_military, run_imap_multiprocessing(calculate, list(chunks(document_ids, 1000)), 10)], axis=0,
                ignore_index=True)
            df_military.replace("", np.nan, inplace=True)
            df_military.dropna(subset=["text"], inplace=True)
            hour_start += timedelta(hours=4)
            hour_end += timedelta(hours=4)
        score_list = []
        dateset = ListDataset(df_military['text'].to_list())
        for out in tqdm(classifier(dateset, label, batch_size=4, gradient_accumulation_steps=16,
                                   gradient_checkpointing=True, fp16=True, optim="adafactor")):
            score_list.append(out['scores'][0])
        df_military['score'] = score_list
        df_military = df_military[df_military['score'] >= 0.7]
        df_military = df_military.drop('text', axis=1)
        df_military["day"] = pd.to_datetime(df_military["day"])
        df_military.to_csv("military_by_day_{}_{}.csv".format(start_date, end_day).replace(" ", "_"), sep="\t",
                           header=True)
        start_date += timedelta(days=1)  # move to next day
        del df_military
        gc.collect()


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27007')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


if __name__ == '__main__':
    # connect to mongo DB
    xlmR_classifier()
    print("--- %s seconds ---" % (time.time() - start_time))
