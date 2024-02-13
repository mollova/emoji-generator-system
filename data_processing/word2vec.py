import random
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import processing_data
import filter_data
import numpy
import time
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity



from operator import itemgetter
from math import sqrt

import spacy
nlp = spacy.load("en_core_web_md")

train_dataset_name = 'datasets/data/train_data_five_emojis.csv'
train_dictionary_filepath = 'datasets/in-progress-data/test-dictionary.dict'
test_dataset_name = 'datasets/data/test_data_five_emojis.csv'
test_dictionary_filepath = 'datasets/in-progress-data/test-dictionary.dict'
five_emojis_dataset_name = 'datasets/data/tweets_with_five_emojis.csv'



train_embeddings_filepath = 'classifier_models/embeddings.pkl'


def squared_sum(x):
  return round(sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = squared_sum(x)*squared_sum(y)
    if denominator == 0:
        return 0.0
    return round(numerator/float(denominator),3)

def save_embedding():
   data_df = processing_data.create_dataframe(dataset_filename=train_dataset_name,
                                               dictionary_path=train_dictionary_filepath,
                                               should_parse=True)
   embeddings = [(nlp(data_df['tweets'][ind]).vector, data_df['emojis'][ind])
                  for ind in data_df.index]

   with open(train_embeddings_filepath, 'wb') as file:
        pickle.dump(embeddings, file)

def load_embedding():
    with open(train_embeddings_filepath, 'rb') as file:
        embeddings = pickle.load(file)

    return embeddings


def jaccard_similarity(x,y):
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

def suggest_emoji_max_jaccard_similarity(input: str):
    data_df = processing_data.create_dataframe(dataset_filename=train_dataset_name,
                                               dictionary_path=train_dictionary_filepath,
                                               should_parse=True)
    tweets = list(data_df.tweets)
    split_tweets = [tweet.lower().split(" ") for tweet in tweets]
    split_input = input.lower().split(" ")

    max_jaccard_sim = 0.0
    max_jaccard_tweet = ""
    for tweet in split_tweets:
        jac = jaccard_similarity(split_input, tweet)
        if jac > max_jaccard_sim:
            max_jaccard_sim = jac
            max_jaccard_tweet = tweet

    emoji = data_df.loc[data_df['tweets'] == " ".join(max_jaccard_tweet)].emojis
    print("Max Jaccard similarity between", split_input, " and ", max_jaccard_tweet, " is: ", max_jaccard_sim)

    return processing_data.integer_to_emoji(list(emoji)[0])

def suggest_emoji_average_jaccard_similarity(input: str):
    average_jaccard_similarities = []
    for dataset_file in filter_data.dataset_emoji_mapping:
        emoji = filter_data.dataset_emoji_mapping[dataset_file]
        data_df = processing_data.create_dataframe(dataset_filename=dataset_file,
                                               dictionary_path=train_dictionary_filepath,
                                               should_parse=True)
        tweets = list(data_df.tweets)
        split_tweets = [tweet.lower().split(" ") for tweet in tweets]
        split_input = input.lower().split(" ")

        jaccard_similarities = [jaccard_similarity(split_input, tweet) for tweet in split_tweets]
        average_jaccard_similarities.append((numpy.average(jaccard_similarities), emoji))

    best_jaccard_tuple = max(average_jaccard_similarities, key = itemgetter(0))
    print("Average best Jaccard similarity is: ", best_jaccard_tuple[0])

    return best_jaccard_tuple[1]


def max_cosine_similarity(input: str):
    embeddings = load_embedding()
    input_embedding = nlp(input).vector

    cos_similarities = [(cos_similarity(input_embedding, embedding[0]), embedding[1])
                        for embedding in embeddings]
    best_tuple = max(cos_similarities, key = itemgetter(0))
    print("Max cosine similarity is: ", best_tuple[0])

    return processing_data.integer_to_emoji(best_tuple[1])


def create_similarity_heatmap_five_tweets():
    data_df = processing_data.create_dataframe(dataset_filename=train_dataset_name,
                                               dictionary_path=train_dictionary_filepath,
                                               should_parse=True)
    tweets = list(data_df.tweets)
    tweets_to_compare = random.sample(tweets, 5)

    similarity = []
    for i in range(len(tweets_to_compare)):
        row = []
        for j in range(len(tweets_to_compare)):
            row.append(cos_similarity(nlp(tweets_to_compare[i]).vector, nlp(tweets_to_compare[i]).vector))
        similarity.append(row)

    # create_heatmap(similarity)
    hm = sns.heatmap(similarity, square=True, annot=True, cbar=False, cmap='red')
    plt.show()

