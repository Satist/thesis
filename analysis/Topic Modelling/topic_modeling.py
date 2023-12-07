""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Performs Topic Modelling
####################################################################################################################"""
import ast
import gc
import os
import re
import time
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
# Plotting tools
import pyLDAvis.gensim_models  # don't skip this
# spacy for lemmatization
import spacy
from gensim.models import CoherenceModel
from gensim.models import HdpModel, LdaModel, LdaMulticore
from gensim.utils import simple_preprocess
# NLTK Stop words
from nltk.corpus import stopwords
from pymongo import MongoClient
import utils
from tqdm.auto import tqdm
from multiprocessing import Pool, pool
from datetime import datetime, timedelta

# import configfile as cnf


stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


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
            new_row = {'user_id': tweet['user']['id'], 'tweet_id': tweet['id'], 'text': text,
                       'created_at': tweet['created_at']}
            df_temp = pd.DataFrame([new_row])
            df_military = pd.concat([df_military, df_temp], axis=0, ignore_index=True)
    return df_military


def start():
    start_date = datetime(2022, 5, 14, 0, 0, 0)
    end_date = datetime(2022, 5, 14, 0, 0, 0)
    while end_date < datetime(2022, 5, 15, 0, 0, 0):
        df_military = pd.DataFrame(columns=['tweet_id', 'text', 'user_id', 'day'])
        end_day = start_date + timedelta(days=1)
        hour_start = start_date
        hour_end = hour_start + timedelta(hours=4)
        print("working on day {} - {}".format(start_date, end_day))
        while hour_end <= end_day:
            print("\t hours {} {}".format(hour_start, hour_end))
            document_ids = calculate_docs(hour_start, hour_end)
            df_military = pd.concat(
                [df_military, run_imap_multiprocessing(calculate, list(chunks(document_ids, 1000)), 4)], axis=0,
                ignore_index=True)
            df_military.replace("", np.nan, inplace=True)
            df_military.dropna(subset=["text"], inplace=True)
            hour_start += timedelta(hours=4)
            hour_end += timedelta(hours=4)
        score_list = []
        lda_generation(df_military, start_date)
        start_date += timedelta(days=1)  # move to next day
        del df_military
        gc.collect()


def connect_to_db():
    clientLoc = MongoClient(host='localhost:27007')
    dbLoc = clientLoc.RussiaWar
    return clientLoc, dbLoc


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield [word for word in
               re.sub('[,\.!?]', '', sentence).lower().replace("\n", " ").replace("\\n", " ").split(" ") if word != ""]
        # yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Defining a function to loop over number of topics to be used to find an
# optimal number of tipics
def compute_coherence_values(id2word, dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values_topic = []
    model_list_topic = []
    maxco = 0
    best_num_topics = 0
    for num_topics in range(start, limit, step):
        start = time.time()
        model = gensim.models.ldamulticore.LdaMulticore(workers=3, corpus=corpus, num_topics=num_topics,
                                                        id2word=id2word)
        model_list_topic.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_topic.append(coherencemodel.get_coherence())
        if maxco < coherencemodel.get_coherence():
            maxco = coherencemodel.get_coherence()
            best_num_topics = num_topics
        end = time.time()
        print("Time: ", end - start)
        print("Num_Topics:", num_topics)

    return model_list_topic, coherence_values_topic, best_num_topics


def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection)) / float(len(union))


def lda_generation(df, cluster_id):
    df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    df['text'] = df['text'].str.encode("ascii", "ignore").str.decode("ascii")
    # print(df["text"])
    data_words = list(sent_to_words(list(df['text'])))
    print(data_words[:1])
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, data_words)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        pprint(doc)
        break

    # View
    print(corpus[:1])

    # Considering 1-50 topics, as the last is cut off
    num_topics = list(range(20)[1:])
    num_keywords = 15

    LDA_models = {}
    LDA_topics = {}
    for i in num_topics:
        print("Testing topic number:", i)
        LDA_models[i] = LdaModel(corpus=corpus_tfidf,
                                 id2word=id2word,
                                 num_topics=i,
                                 chunksize=len(corpus_tfidf),
                                 passes=20,
                                 alpha='auto',
                                 random_state=42,
                                 )

        shown_topics = LDA_models[i].show_topics(num_topics=i,
                                                 num_words=num_keywords,
                                                 formatted=False)
        LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
    LDA_stability = {}
    for i in range(0, len(num_topics) - 1):
        print("Testing topic number for jaccard-simms:", i)
        jaccard_sims = []
        for t1, topic1 in enumerate(LDA_topics[num_topics[i]]):  # pylint: disable=unused-variable
            sims = []
            for t2, topic2 in enumerate(LDA_topics[num_topics[i + 1]]):  # pylint: disable=unused-variable
                sims.append(jaccard_similarity(topic1, topic2))

            jaccard_sims.append(sims)

        LDA_stability[num_topics[i]] = jaccard_sims

    mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
    coherences = [
        CoherenceModel(model=LDA_models[i], texts=texts, dictionary=id2word, coherence='c_v').get_coherence() \
        for i in num_topics[:-1]]
    coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in
                     range(num_keywords)[:-1]]  # limit topic numbers to the number of keywords
    coh_sta_max = max(coh_sta_diffs)
    coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    ideal_topic_num_index = coh_sta_max_idxs[0]  # choose less topics in case there's more than one max
    ideal_topic_num = num_topics[ideal_topic_num_index]
    optimalldamodel = LdaModel(corpus=corpus_tfidf,
                               id2word=id2word,
                               num_topics=ideal_topic_num,
                               chunksize=len(corpus_tfidf),
                               passes=20,
                               alpha='auto',
                               random_state=42,
                               )
    vis = pyLDAvis.gensim_models.prepare(optimalldamodel, corpus_tfidf, id2word, mds='mmds')
    pyLDAvis.save_html(vis, 'topics_by_date{}.html'.format(cluster_id))


if __name__ == '__main__':
    start()
    # close connection to mongoDB
