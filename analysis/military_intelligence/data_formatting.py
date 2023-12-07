from __future__ import unicode_literals, print_function

import collections
from itertools import groupby

import numpy as np
import plac
import random
from pathlib import Path
import spacy
from sklearn.model_selection import train_test_split
from spacy.util import minibatch, filter_spans
from tqdm import tqdm
import pandas as pd
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from googletrans import Translator
translator = Translator()


def combine_item_pairs(l1, l2):
    D = {k: [v, False] for k, v in l1}
    for key, value in l2:
        if key in D:
            D[key][1] = value
        else:
            D[key] = [False, value]
    return (tuple([key] + value) for key, value in D.items())


def split_df(df):
    df = df.reset_index()
    l1 = []
    l2 = []
    for i in range(0, len(df['text'])):
        l1.append(df['text'][i])
        l2.append([(int(df['begin'][i]), int(df['end'][i]), df['type'][i])])
    TRAIN_DATA = list(zip(l1, l2))
    c = collections.defaultdict(list)
    for a, b in TRAIN_DATA:
        c[a].extend(b)  # add to existing list or create a new one

    list_2 = list(c.items())
    return list_2


def docbin(list):
    db = DocBin()
    nlp = spacy.blank("xx")
    for text, annot in tqdm(list):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot:  # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        filtered = filter_spans(ents)
        doc.ents = filtered  # label the text with the ents
        db.add(doc)
    return db


if __name__ == '__main__':
    df = pd.read_csv("dataset/train.csv")
    df = df[df['value'] != ' ']
    df = df.reset_index()
    df['doc']
    list = split_df(df)
    training_data, testing_data = train_test_split(list, test_size=0.2, random_state=25)
    training_data = docbin(training_data)
    testing_data = docbin(testing_data)
    training_data.to_disk("./train.spacy")
    testing_data.to_disk("./dev.spacy")
