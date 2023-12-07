import re
import string

import emoji
import math
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

exclude = set(string.punctuation).union(set(["'", '`', "’", "\""])) - set(["#", "@", "_"])
stop_words = set(stopwords.words('english'))

"""
remove double spaces from text
"""


def remove_double_space(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text


"""
Filter the tweet text from non-necessary characters 
"""


def filter_the_text(text):
    text = text.replace("\\n", " ").replace("…", " ").replace("...", " ")
    text = re.sub('[\n\t:]', ' ', text)

    return remove_double_space(text)


"""
get twitter text and removes the mentions from it. 
The filtered text and the list of extracted mentions are returned 
"""


def remove_mentions(text):
    mentions = [w for w in text.split(" ") if w != '' and w[0] == "@"]
    text = " ".join([x for x in text.split(" ") if x != '' and x[0] != "@"])
    return mentions, text


def remove_hashtags(text):
    hashtags = [w for w in text.split(" ") if w != '' and w[0] == "#"]
    text = " ".join([x for x in text.split(" ") if x != '' and x[0] != "#"])

    return hashtags, text


def remove_punctuations(s):
    return remove_double_space(''.join(ch for ch in s if ch not in exclude))


def remove_stop_words(s, keep_case=False):
    global stop_words

    words = " ".join([word for word in remove_double_space(s).split(" ") if word.lower() not in stop_words])

    result = []

    temp = []
    for i in sent_tokenize(words):

        for j in word_tokenize(remove_punctuations(i)):
            # since word tokenizer will also separate mentions/hashtags characters
            # we need to combine them back together
            if len(temp) > 0 and temp[-1] == "#":
                temp[-1] = "#" + j
            elif len(temp) > 0 and temp[-1] == "@":
                temp[-1] = "@" + j
            else:
                temp.append(j if keep_case else j.lower())

    for item in temp:
        if item not in string.punctuation \
                and not item.isnumeric() \
                and " " not in item \
                and item != '' \
                and len(item) > 1:
            result.append(item)
    return result


"""
Remove emojis from text. List of emojis and clear text is returned
"""


def extract_emojis(text):
    emojis = [c for c in text if c in emoji.UNICODE_EMOJI]
    for emj in emojis:
        text = text.replace(emj, " ")

    return emojis, remove_double_space(text)


def extract_urls(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = [x[0] for x in re.findall(regex, text)]
    for link in url:
        text = text.replace(link, " ")

    return url, remove_double_space(text)


def fix_mentions_hashtags(text):
    # fix collated hashtags
    text = text.replace("#", " #").replace("@", " @")

    # remove empty mentions/hashtags and also remove multiple spaces/tabs/new lines
    text = filter_the_text(text.replace("# ", " ").replace("@ ", " "))
    return text


def make_all(text, ent, keep_case=False):
    # Remove URL links from text
    # print("Make all before extract urls:{}".format(text))
    urls, text = extract_urls(text)
    # print("Make all after url extraction:{}".format(text))

    # fix wrong mentions/hashtags
    text = fix_mentions_hashtags(text)

    # Remove emojis
    emjis, text = extract_emojis(text)

    # remove RT/dots and stop words from text
    text = text.replace(".", " ")
    if text.startswith("RT"):
        text = text[2:]
    text = [x for x in remove_stop_words(filter_the_text(text), keep_case)]

    # add hashtags that appears in tweet object but they are not in tweet text
    for hs in ent["hashtags"]:
        if "#" + hs["text"] not in text:
            text.append("#{}".format(hs["text"]))

    # add mentions that appears in tweet object but they are not in tweet text
    for mnt in ent["user_mentions"]:
        if "@" + mnt["screen_name"] not in text:
            text.append("@{}".format(mnt["screen_name"]))
    return text, emjis


"""
Merge the original tweet text and retweet from Twitter object, in order to get full text
"""


def merge_tw_rt(tweet_text, retweet_text):
    if ": " not in tweet_text:
        if "account is temporarily unavailable because it violates the Twitter Media Policy. Learn more" in tweet_text:
            return None
        else:
            print("Tweet:->{}\nRetweet:->{}\n\n".format(tweet_text, retweet_text))
            ind = 0
    else:
        ind = tweet_text.index(": ") + 2
    start_ind_tw = None
    start_ind_rt = None
    # ind = tw.index(": ")+ 2
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

    if "…" in tweet_text or len(new) > len(tweet_text):
        return new
    return tweet_text


"""
Merge entities of tweet object from tweet and retweet fileds, if there any differences
"""


def merge_entities(tw_ent, rt_ent):
    ent = {"hashtags": [], "user_mentions": [], "urls": []}

    # store tweet and retweet hashtag entitys in ent dictionary
    for hs in tw_ent["hashtags"] + rt_ent["hashtags"]:
        entry = {"text": hs["text"]}
        if entry not in ent["hashtags"]:
            ent["hashtags"].append(entry)

    # store tweet and retweet mention entitys in ent dictionary
    for hs in tw_ent["user_mentions"] + rt_ent["user_mentions"]:
        entry = {"screen_name": hs["screen_name"]}
        if entry not in ent["user_mentions"]:
            ent["user_mentions"].append(entry)

    # store url link from retweet and tweet object
    for url in tw_ent["urls"] + rt_ent["urls"]:
        entry = {"url": url["url"]}
        if entry not in ent["urls"]:
            ent["urls"].append(entry)
    return ent


def compute_std(data):
    if len(data) == 0:
        return 0.0
    varience = sum([x ** 2 for x in data]) / len(data)
    return math.sqrt(varience)


def text_feature(text, data, flag, ent):
    text, emojis = make_all(text, ent, keep_case=True)
    # print("after make all the text is {}\n\n".format(text))
    hs = [x for x in text if len(x) > 1 and x[0] == "#"]
    mnt = [x for x in text if len(x) > 1 and x[0] == "@"]

    if flag == "pre":
        return hs, mnt

    text = [x for x in text if len(x) > 0 and x[0] not in ["#", "@"]]
    for word in text:
        # print("Word:{}".format(word))
        if word[0].isupper():
            data["upper_frq_{}".format(flag)][word] += 1
        data["words_frq_{}".format(flag)][word.lower()] += 1

    # if flag == "rt":
    #  print("---------------")
    #  print(data["words_frq_rt"])
    #  print("+++++++ END ++++++\n\n")
    # for emj in emojis:
    #  data["emj_frq_{}".format(flag)][emj] += 1
    for hashtag in hs:
        data["hash_" + flag][hashtag] += 1
    for mention in mnt:
        data["ment_" + flag][mention] += 1
