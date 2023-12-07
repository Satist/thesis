""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Performs Clustering
####################################################################################################################"""

import os

import pandas as pd

from utils import *

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

if __name__ == '__main__':
    iterations = 100
    data = pd.read_csv('post_features_suspended_only.csv', sep='\t')

    ids = data["tweet_id"].copy()
    data.drop(["text_category", "tweet_id", "user_id", "created_at"], axis=1, inplace=True)

    data.reset_index(drop=True, inplace=True)
    ids.reset_index(drop=True, inplace=True)
    emb = {ids.iloc[i]: data.iloc[i, :].to_list() for i in data.index.to_list()}
    del data

    clusters = {}
    clusters = online_community_detection(ids, emb, clusters, chunk_size=5000, cores=24, iterations=iterations)

    f_out = open("text_clusters_{}.txt".format(iterations), "w+")
    for cluster in list(clusters.values()):
        f_out.write("{}\n".format(cluster))
    f_out.close()
