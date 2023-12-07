""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Performs Toxicity Labeling using the Detoxify multilingual library
####################################################################################################################"""
import ast
import gc
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from detoxify import Detoxify
from pymongo import MongoClient
from tqdm import tqdm
# General
from tqdm.auto import tqdm

import utils

# Asthetics

pd.set_option('display.max_columns', None)
np.seterr(divide='ignore', invalid='ignore')
gc.enable()
tqdm.pandas()


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


def get_df(users):
    client, db = connect_to_db()
    df_mongo = pd.DataFrame(columns=['user_id', 'tweet_id', 'text'])
    tweet = db.Tweets.find({"user.id": users},
                           {'id': 1, 'user': 1, 'text': 1, 'extended_tweet': 1, 'retweeted_status': 1})
    tweet = list(tweet)
    for j in tweet:
        text = get_text(j)
        if text is not None:
            new_row = {'user_id': j['user']['id'], 'tweet_id': j['id'], 'text': text}
            df_temp = pd.DataFrame([new_row])
            df_mongo = pd.concat([df_mongo, df_temp], axis=0, ignore_index=True)
    return df_mongo


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)

    result_list_tqdm = pd.DataFrame(columns=['user_id', 'tweet_id', 'text'])
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm = pd.concat([result_list_tqdm, result], axis=0, ignore_index=True)

    return result_list_tqdm


def get_multiple_tweets(users):
    print("\n\nMultiple tweets extraction:")
    threshold = 0.75
    df_mongo = run_imap_multiprocessing(get_df, users, 20)
    df_mongo['text'] = df_mongo['text'].swifter.progress_bar(True).apply(text_cleaning)

    start_time = time.time()
    results = df_mongo['text'].swifter.progress_bar(True).apply(Detoxify('multilingual', device='cuda').predict)
    results = pd.DataFrame(results.tolist())
    results.loc[results["toxicity"] < threshold, "toxicity"] = np.nan
    results.loc[results["severe_toxicity"] < threshold, "severe_toxicity"] = np.nan
    results.loc[results["identity_attack"] < threshold, "identity_attack"] = np.nan
    results.loc[results["insult"] < threshold, "insult"] = np.nan
    results.loc[results["threat"] < threshold, "threat"] = np.nan
    results.loc[results["sexual_explicit"] < threshold, "sexual_explicit"] = np.nan
    results.loc[results["obscene"] < threshold, "obscene"] = np.nan
    results = results.dropna(how='all')
    if not results.empty:
        results["user_id"] = df_mongo['user_id']
        results["tweet_id"] = df_mongo['tweet_id']

        results.to_csv("/media/disk1/toxicity_results/toxicity.csv", na_rep='0')
    print("Total Time: ", time.time() - start_time, "s")


def load_users():
    path = "compliance.txt"  # Path to complience file or another file that contain suspended accounts
    # read compliance file and keep suspended user ids
    suspend_users = [int(ast.literal_eval(x)["id"]) for x in
                     os.popen("cat {} | grep suspend".format(path)).read().split("\n")[:-1]]
    return suspend_users


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27017')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


if __name__ == '__main__':
    susp_users = load_users()
    get_multiple_tweets(susp_users)
