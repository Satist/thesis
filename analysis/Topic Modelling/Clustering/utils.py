""""####################################################################################################################
Author: Ioannis Lamprou ICS-FORTH / CSD-UOC
E-mail: csd3976@csd.uoc.gr
-----------------------------------
Contains the functions used in clustering.py
####################################################################################################################"""

import logging
import math
import os
from collections import defaultdict

import numpy as np
import pickle5 as pickle
import torch
from funcy import log_durations
from joblib import Parallel
from joblib import delayed
from torch import Tensor
from tqdm import tqdm


def cos_sim(a: Tensor, b: Tensor):
    """
  Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
  :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
  """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(np.array(a))

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(np.array(b))

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def get_embeddings(ids, embeddings):
    return np.array([embeddings[idx] for idx in ids])


def reorder_and_filter_cluster(
        cluster_idx, cluster, cluster_embeddings, cluster_head_embedding, threshold
):
    cos_scores = cos_sim(cluster_head_embedding, cluster_embeddings)
    sorted_vals, indices = torch.sort(cos_scores[0], descending=True)
    bigger_than_threshold = sorted_vals > threshold
    indices = indices[bigger_than_threshold]
    sorted_vals = sorted_vals.numpy()
    return cluster_idx, [(cluster[i][0], sorted_vals[i]) for i in indices]


def get_ids(cluster):
    return [transaction[0] for transaction in cluster]


def reorder_and_filter_clusters(clusters, embeddings, threshold, parallel):
    results = parallel(
        delayed(reorder_and_filter_cluster)(
            cluster_idx,
            cluster,
            get_embeddings(get_ids(cluster), embeddings),
            get_embeddings([cluster_idx], embeddings),
            threshold,
        )
        for cluster_idx, cluster in tqdm(clusters.items())
    )

    clusters = {k: v for k, v in results}

    return clusters


def get_embeddings(ids, embeddings):
    return np.array([embeddings[idx] for idx in ids])


def get_clustured_ids(clusters):
    clustered_ids = set(
        [transaction[0] for cluster in clusters.values() for transaction in cluster]
    )
    clustered_ids |= set(clusters.keys())
    return clustered_ids


def get_clusters_ids(clusters):
    return list(clusters.keys())


def get_unclustured_ids(ids, clusters):
    clustered_ids = get_clustured_ids(clusters)
    unclustered_ids = list(set(ids) - clustered_ids)
    return unclustered_ids


def sort_clusters(clusters):
    return dict(
        sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    )  # sort based on size


def sort_cluster(cluster):
    return list(
        sorted(cluster, key=lambda x: x[1], reverse=True)
    )  # sort based on similarity


def filter_clusters(clusters, min_cluster_size):
    return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}


def unique(collection):
    return list(dict.fromkeys(collection))


def unique_txs(collection):
    seen = set()
    return [x for x in collection if not (x[0] in seen or seen.add(x[0]))]


def write_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def chunk(txs, chunk_size):
    n = math.ceil(len(txs) / chunk_size)
    k, m = divmod(len(txs), n)
    return (txs[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def online_community_detection(
        ids,
        embeddings,
        clusters=None,
        threshold=0.7,
        min_cluster_size=3,
        chunk_size=2500,
        iterations=10,
        cores=10,
):
    if clusters is None:
        clusters = {}

    with Parallel(n_jobs=cores) as parallel:
        for iteration in range(iterations):
            print("1. Nearest cluster")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            cluster_ids = list(clusters.keys())
            print("Unclustured", len(unclustered_ids))
            print("Clusters", len(cluster_ids))
            clusters = nearest_cluster(
                unclustered_ids,
                embeddings,
                clusters,
                chunk_size=chunk_size,
                parallel=parallel,
            )
            print("\n\n")

            print("2. Create new clusters")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            print("Unclustured", len(unclustered_ids))
            new_clusters = create_clusters(
                unclustered_ids,
                embeddings,
                clusters={},
                min_cluster_size=3,
                chunk_size=chunk_size,
                threshold=threshold,
                parallel=parallel,
            )
            new_cluster_ids = list(new_clusters.keys())
            print("\n\n")

            print("3. Merge new clusters", len(new_cluster_ids))
            max_clusters_size = 25000
            while True:
                new_cluster_ids = list(new_clusters.keys())
                old_new_cluster_ids = new_cluster_ids
                new_clusters = create_clusters(
                    new_cluster_ids,
                    embeddings,
                    new_clusters,
                    min_cluster_size=1,
                    chunk_size=max_clusters_size,
                    threshold=threshold,
                    parallel=parallel,
                )
                new_clusters = filter_clusters(new_clusters, 2)

                new_cluster_ids = list(new_clusters.keys())
                print("New merged clusters", len(new_cluster_ids))
                if len(old_new_cluster_ids) < max_clusters_size:
                    break

            new_clusters = filter_clusters(new_clusters, min_cluster_size)
            print(
                f"New clusters with min community size >= {min_cluster_size}",
                len(new_clusters),
            )
            clusters = {**new_clusters, **clusters}
            print("Total clusters", len(clusters))
            clusters = sort_clusters(clusters)
            print("\n\n")

            print("4. Nearest cluster")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            cluster_ids = list(clusters.keys())
            print("Unclustured", len(unclustered_ids))
            print("Clusters", len(cluster_ids))
            clusters = nearest_cluster(
                unclustered_ids,
                embeddings,
                clusters,
                chunk_size=chunk_size,
                parallel=parallel,
            )
            clusters = sort_clusters(clusters)

            unclustered_ids = get_unclustured_ids(ids, clusters)
            clustured_ids = get_clustured_ids(clusters)
            print("Clustured", len(clustured_ids))
            print("Unclustured", len(unclustered_ids))
            print(
                f"Percentage clustured {len(clustured_ids) / (len(clustured_ids) + len(unclustered_ids)) * 100:.2f}%"
            )

            print("\n\n")
    return clusters


def get_ids(cluster):
    return [transaction[0] for transaction in cluster]


def nearest_cluster_chunk(
        chunk_ids, chunk_embeddings, cluster_ids, cluster_embeddings, threshold
):
    cos_scores = cos_sim(chunk_embeddings, cluster_embeddings)
    top_val_large, top_idx_large = cos_scores.topk(k=1, largest=True)
    top_idx_large = top_idx_large[:, 0].tolist()
    top_val_large = top_val_large[:, 0].tolist()
    cluster_assignment = []
    for i, (score, idx) in enumerate(zip(top_val_large, top_idx_large)):
        cluster_id = cluster_ids[idx]
        if score > threshold:
            cluster_assignment.append(((chunk_ids[i], score), cluster_id))
    return cluster_assignment


def nearest_cluster(
        transaction_ids,
        embeddings,
        clusters=None,
        parallel=None,
        threshold=0.75,
        chunk_size=2500,
):
    cluster_ids = list(clusters.keys())
    if len(cluster_ids) == 0:
        return clusters
    cluster_embeddings = get_embeddings(cluster_ids, embeddings)

    c = list(chunk(transaction_ids, chunk_size))

    with log_durations(logging.info, "Parallel jobs nearest cluster"):
        out = parallel(
            delayed(nearest_cluster_chunk)(
                chunk_ids,
                get_embeddings(chunk_ids, embeddings),
                cluster_ids,
                cluster_embeddings,
                threshold,
            )
            for chunk_ids in tqdm(c)
        )
        cluster_assignment = [assignment for sublist in out for assignment in sublist]

    for (transaction_id, similarity), cluster_id in cluster_assignment:
        if cluster_id is None:
            continue
        clusters[cluster_id].append(
            (transaction_id, similarity)
        )

    clusters = {
        cluster_id: unique_txs(sort_cluster(cluster))
        for cluster_id, cluster in clusters.items()
    }  # Sort based on similarity

    return clusters


def create_clusters(
        ids,
        embeddings,
        clusters=None,
        parallel=None,
        min_cluster_size=3,
        threshold=0.75,
        chunk_size=2500,
):
    to_cluster_ids = np.array(ids)
    np.random.shuffle(
        to_cluster_ids
    )

    c = list(chunk(to_cluster_ids, chunk_size))

    with log_durations(logging.info, "Parallel jobs create clusters"):
        out = parallel(
            delayed(fast_clustering)(
                chunk_ids,
                get_embeddings(chunk_ids, embeddings),
                threshold,
                min_cluster_size,
            )
            for chunk_ids in tqdm(c)
        )

    # Combine output
    new_clusters = {}
    for out_clusters in out:
        for idx, cluster in out_clusters.items():
            # new_clusters[idx] = unique([(idx, 1)] + new_clusters.get(idx, []) + cluster)
            new_clusters[idx] = unique_txs(cluster + new_clusters.get(idx, []))

    # Add ids from old cluster to new cluster
    for cluster_idx, cluster in new_clusters.items():
        community_extended = []
        for (idx, similarity) in cluster:
            community_extended += [(idx, similarity)] + clusters.get(idx, [])
        new_clusters[cluster_idx] = unique_txs(community_extended)

    new_clusters = reorder_and_filter_clusters(
        new_clusters, embeddings, threshold, parallel
    )  # filter to keep only the relevant
    new_clusters = sort_clusters(new_clusters)

    clustered_ids = set()
    for idx, cluster_ids in new_clusters.items():
        filtered = set(cluster_ids) - clustered_ids
        cluster_ids = [
            cluster_idx for cluster_idx in cluster_ids if cluster_idx in filtered
        ]
        new_clusters[idx] = cluster_ids
        clustered_ids |= set(cluster_ids)

    new_clusters = filter_clusters(new_clusters, min_cluster_size)
    new_clusters = sort_clusters(new_clusters)
    return new_clusters


def fast_clustering(ids, embeddings, threshold=0.70, min_cluster_size=10):
    """
  Function for Fast Clustering

  Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
  """

    # Compute cosine similarity scores
    cos_scores = cos_sim(embeddings, embeddings)

    # Step 1) Create clusters where similarity is bigger than threshold
    bigger_than_threshold = cos_scores >= threshold
    indices = bigger_than_threshold.nonzero()

    cos_scores = cos_scores.numpy()

    extracted_clusters = defaultdict(lambda: [])
    for row, col in indices.tolist():
        extracted_clusters[ids[row]].append((ids[col], cos_scores[row, col]))

    extracted_clusters = sort_clusters(extracted_clusters)

    # Step 2) Remove overlapping clusters
    unique_clusters = {}
    extracted_ids = set()

    for cluster_id, cluster in extracted_clusters.items():
        add_cluster = True
        for transaction in cluster:
            if transaction[0] in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_clusters[cluster_id] = cluster
            for transaction in cluster:
                extracted_ids.add(transaction[0])

    new_clusters = {}
    for cluster_id, cluster in unique_clusters.items():
        community_extended = []
        for idx in cluster:
            community_extended.append(idx)
        new_clusters[cluster_id] = unique_txs(community_extended)

    new_clusters = filter_clusters(new_clusters, min_cluster_size)

    return new_clusters


def merge_tw_rt(tweet_text, retweet_text):
    if ": " not in tweet_text:
        print("Tweet:->{}\nRetweet:->{}\n\n".format(tweet_text, retweet_text))
        ind = 0
    else:
        ind = tweet_text.index(": ") + 2

    start_ind_tw = None
    start_ind_rt = None

    if len(tweet_text) - ind <= 6:
        if tweet_text[ind:] in retweet_text:
            start_ind_rt = retweet_text.index(tweet_text[ind:])
            start_ind_tw = ind
    else:
        for i in range(ind, len(tweet_text) - 6):
            if tweet_text[i:i + 6] in retweet_text:
                start_ind_rt = retweet_text.index(tweet_text[i:i + 6])
                start_ind_tw = i
                break
    if start_ind_tw != None:
        new = tweet_text[:start_ind_tw] + retweet_text[start_ind_rt:]
    else:
        print("No intersection between tw:{}\nrt:{}\n".format(tweet_text, retweet_text))
        new = tweet_text

    if "…" not in tweet_text and len(new) <= len(tweet_text):
        new = tweet_text
    return new
