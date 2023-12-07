import datetime
import glob
import os
import re
import time
from multiprocessing import Pool
import swifter
import pandas as pd
from bs4 import BeautifulSoup
from pymongo import MongoClient
from tqdm import tqdm

import spacy
import utils


spacy.prefer_gpu()
nlp = spacy.load("/home/glamprou/spacy/output/model-last/")  # load the model


def RemoveBannedWords(toPrint):
    database = ['#Ukraine', '#Ukraina', '#ukraina', '#Украина', '#Украине', '#PrayForUkraine', '#UkraineRussie',
                '#StandWithUkraine', '#StandWithUkraineNOW', '#RussiaUkraineConflict', '#RussiaUkraineCrisis',
                '#RussiaInvadedUkraine', '#WWIII', '#worldwar3', '#Война', '#BlockPutinWallets',
                '#UkraineRussiaWar', '#Putin', '#Russia', '#Россия', '#StopPutin', '#StopRussianAggression',
                '#StopRussia', '#Ukraine Russia', '#Russian Ukrainian', '#FuckPutin', '#solidarityWithUkraine',
                '#PutinWarCriminal', '#PutinHitler', '#BoycottRussia', '#with russia',
                '#FUCK NATO', '#ЯпротивВойны', '#StopNazism', '#myfriendPutin', '#UnitedAgainstUkraine',
                '#StopWar', '#ВпередРоссия', '#ЯМыРоссия', '#ВеликаяРоссия', '#Путинмойпрезидент',
                '#россиявперед', '#россиявперёд', '#ПутинНашПрезидент', '#ЗаПутина', '#Путинмойпрезидент',
                '#ПутинВведиВойска', '#СЛАВАРОССИИ', '#СЛАВАВДВ']
    return ' '.join(i for i in toPrint.split() if i not in database)


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


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27007')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


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


def extract_labels(text):
    doc = nlp(text)
    results = [(ent.text, ent.label_) for ent in doc.ents]
    return results


def get_multiple_tweets():
    path = "/home/glamprou/spacy/csv"
    csv_files = glob.glob(path + "/*.csv")
    for file_name in csv_files:
        x = pd.read_csv(file_name, sep='\t')
        print(x.head(10))
        df_mongo = run_imap_multiprocessing(calculate, list(chunks(x['tweet_id'].to_list(), 1000)), 10)
        start_time = time.time()
        df_mongo["entities"] = df_mongo['text'].astype(str).swifter.progress_bar(True).apply(extract_labels)
        df_mongo = df_mongo.drop('text', axis=1)
        df_mongo.to_csv("/home/glamprou/ner_results/ner_results_{}.csv".format(df_mongo['created_at'].iloc[0]), na_rep='0')
        print("Total Time: ", time.time() - start_time, "s")


if __name__ == '__main__':
    get_multiple_tweets()
