import warnings
from datetime import datetime
from os import path

import gensim
from gensim.models import Word2Vec

from utils import *

warnings.filterwarnings(action='ignore')


class WordToVec:
    def __init__(self, month, filename="word2vec_suspension.model", verbose=False):
        self.mongo = mongoClass()
        self.end_date = datetime(2020, 9 + month, 1)
        self.start_date = datetime(2020, 8 + month, 1)
        self.model = None
        self.data = []
        self.filename = filename.replace(".model", "_month_{}.model".format(8 + month))
        self.make_word2vec(verbose=verbose)

    def collect(self):
        iter_cnt = 0
        dates_filter = {"created_at": {"$lt": self.end_date, "$gte": self.start_date}}
        all_size = self.mongo.count_tweets(filter=dates_filter)
        timer = TimeMeasure(all_size)
        # all_size = db.tweets.count({"created_at": {"$lt": end_date, "$gte": start_date}})
        for tw in self.mongo.get_tweets(filter=dates_filter):
            iter_cnt += 1

            # Print iteration information
            if iter_cnt % 500000 == 0:
                spend_time, left_time = timer.measure_time(iter_cnt, "hours")
                print(
                    "Iteration :{} of :{}. Time spend:{} h. and time left:{} h.".format(iter_cnt, all_size, spend_time,
                                                                                        left_time))

            # get twitter post text from object (if full_text exist use it else use just text field)
            tweet_text = tw["full_text"] if "full_text" in tw else tw["text"]
            if "account is temporarily unavailable because it violates the Twitter Media Policy." in tweet_text:
                # ignore post it is not available
                continue

            # In case when retweeted status exist, required merge of text and entities
            if "retweeted_status" in tw:
                tweet_text = merge_tw_rt(tweet_text,
                                         tw["retweeted_status"]["full_text"] if "full_text" in
                                                                                tw["retweeted_status"] else
                                         tw["retweeted_status"]["text"])

                if tweet_text == None:
                    # ignore post it is not available
                    continue

                entities = merge_entities(tw["entities"], tw["retweeted_status"]["entities"])
            else:
                entities = tw["entities"]

            tweet_text, _ = make_all(tweet_text, entities)

            if len(tweet_text) > 0:
                tweet_text = [word for word in tweet_text if word != ""]
                self.data.append(tweet_text)
        self.mongo.close()

    """
    Train model based on collected data
    """

    def train_model(self):
        self.model = gensim.models.Word2Vec(self.data, min_count=1,
                                            size=10, window=5)

    """
    Store trained model
    """

    def save_model(self):
        self.model.save(self.filename)

    def make_word2vec(self, verbose=False):
        # check if file model file is already exist
        if path.exists(self.filename):
            print("Word2Vec model already exist.")
            return None

        if verbose:
            print("Start the tweets collection of Word2Vec model")
        self.collect()

        if verbose:
            print("Train of the model")
        self.train_model()

        if verbose:
            print("Store model in: {} file".format(self.filename))
        self.save_model()
