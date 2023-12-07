import ast
from multiprocessing import Pool

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
# General
from tqdm.auto import tqdm
import utils


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27007')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


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


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)

    result_list_tqdm = pd.DataFrame(columns=['user_id', 'text'])
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm = pd.concat([result_list_tqdm, result], axis=0, ignore_index=True)

    return result_list_tqdm


def get_df(tweet_id):
    client, db = connect_to_db()
    df_mongo = pd.DataFrame(columns=['user_id', 'text'])
    tweet = db.Tweets.find({"id": tweet_id},
                           {'id': 1, 'user': 1, 'text': 1, 'extended_tweet': 1, 'retweeted_status': 1})
    tweet = list(tweet)
    for j in tweet:
        text = get_text(j)
        if j['user']['id'] not in df_mongo['user_id']:
            new_row = {'user_id': j['user']['id'], 'text': text}
            df_temp = pd.DataFrame([new_row])
            df_mongo = pd.concat([df_mongo, df_temp], axis=0, ignore_index=True)
    return df_mongo


if __name__ == '__main__':
    data = []
    df_ids = pd.DataFrame(columns=['user_id', 'cluster'])
    with open("/home/johnlamprou/PycharmProjects/TwitterSuspension/analysis/Toxicity/new_clusters_with_iter85.txt", "r") as inFile:
        data = ast.literal_eval(inFile.read())
    i = 0
    for key, value in tqdm(data.items()):
        tweet_ids = list(zip(*value))[0]
        df_mongo = run_imap_multiprocessing(get_df, tweet_ids, 8)
        df_mongo.to_csv("cluster_{}.csv".format(i))
        i = i + 1
