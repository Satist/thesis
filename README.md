# Analysis of social media usage during the Russo-Ukrainian war (BSc Thesis)



## Abstract

On February 24, 2022, Russia’s invasion of Ukraine, now
known as the Russo-Ukrainian War, sparked extensive discussions on Online Social Networks (OSN). To capture this
dynamic environment, including analyzing the discussed topics and detecting potential malicious activities, we initiated
an ongoing data collection effort using the Twitter API. As
of the writing of this paper, our dataset comprises 119.6 million tweets from 10.4 million users. Given the dataset’s diverse linguistic composition and the absence of labeled data,
we approached it as a zero-shot learning problem, employing
various techniques that required no prior supervised training
on the dataset.
Our research covers several areas, including sentiment analysis to gauge the public response to the distressing events of the
war, topic analysis to compare narratives between social media and traditional media, and an examination of message toxicity levels, which has led to increased Twitter suspensions.
Additionally, we explore the potential exploitation of social
media for acquiring military-related information by belligerents, presenting a novel machine-learning methodology for
classifying such communications.
The findings of this study provide fresh insights into the role
of social media during conflicts, with broad implications for
policy, security, and information dissemination.




## Authors

- [@Ioannis Lamprou](https://www.github.com/jlamprou)
- [@Alexander Shevtsov](https://github.com/alexdrk14)



## Preprocessing
    Preprocessing of tweet include filtering/removing the following points:
    * Any URLs (e.g. www.xyz.com) for all analysis
    * Extra spaces in the text
    * Hashtags(#topic) and usernames(@user) for Sentiment analysis and Toxicity Analysis
    * Emoticons except in sentiment analysis.

## Sentiment Analysis
    Using The multilingual version of RoBERTa->XLM-RoBERTa. 100 languages from
    2.5TB of filtered CommonCrawl data used as its pre-training material. The labels that we
    used for the model were positive, neutral, and negative for Ukraine, Russia, Zelensky and
    Putin. Thanks to our GPU implementation, this methodology allows the process of a large
    amount of data in order to provide daily sentiment values.

## Topic Modelling
    Using BERTopic:
    1. Embeddings: We start by converting our documents to vector representations through
    the use of language models.
    2. Dimension Reduction: The vector representations are reduced in dimensionality so
    that the clustering algorithms have an easier time finding clusters. (UMAP [11],PCA).
    3. Clustering: A clustering algorithm is used to cluster the reduced vectors in order to
    find semantically similar ones. documents.(HDBSCAN [10],k-Means,BIRCH)
    4. Bag of Words: We tokenize each topic into a bag-of-words representation that allows
    us to process the data without affecting the input embeddings (CountVectorizer).
    5. Topic Representation: We calculate words that are related to each topic with a classbased TF-IDF procedure called c-TF-IDF.

## Toxicity Analysis
    Using for toxic comment classification Detoxify, a state-of-art model,
    pre-trained in social media datasets and it can classify multilingual corpus. After researching the field of toxicity classification methods, we decided that for our multilingual corpus,
    this model would provide the best accuracy and performance without custom training.

## Military Intelligence
    Using as a base spaCy’s NER model and train it with the only open-source military datase by the Defence Science and Technology Laboratory of the UK. 
    We create an algorithm to filter our dataset and extract Military entities :
    1. XLM-RoBERTa model for zero-shot classification using the label "military" with a
    threshold > 0.7 (range 0-1)
    2. Use our NER model to extract entities
    3. Filter location entities per tweet for Ukrainian locations.
    Using the extracted tweets and entities we can perform data analysis and statistics for military events and information daily for any Ukrainian location.

## Clustering
    Using the sentence embeddings of the Tweets we calculate the cosine similarity of the Tweets. The computation of the clusters is implemented based on a clustering solution from (https://ntropy.com/post/clustering-millions-of-sentences-to-optimize-the-ml-workflow) , they identify the centers and compute the distance between the unclustered elements and each center of the cluster. This particular implementation provides the cluster centers, in order to identify the text and the topic of each cluster
