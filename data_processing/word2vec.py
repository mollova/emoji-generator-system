import processing_data
import filter_data
import numpy
import time
import pickle

from operator import itemgetter
from math import sqrt

import spacy
nlp = spacy.load("en_core_web_md")

train_dataset_name = 'datasets/data/tweets_with_five_emojis.csv'
train_dictionary_filepath = 'datasets/in-progress-data/nlp/dictionary.dict'

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
    
# save_embedding()
max_cosine_similarity("I love you")


# suggest_emoji_max_jaccard_similarity("I love you")
# print()
# suggest_emoji_average_jaccard_similarity("I love you")








# import processing_data
# import nltk
# import numpy as np

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# cbow_model_filepath = 'classifier_models/nlp/cbow.pkl'

# # def train_cbow():
# clf, doc_model, data = processing_data.train_model_CBOW()
#     # processing_data.save_trained_model(clf, cbow_model_filepath)
# # import gensim
# # a = gensim.models.KeyedVectors()
# # a.wm

# # train_cbow()

# # model, _ = processing_data.load_trained_model(cbow_model_filepath)
# # print("Cosine similarity between 'love' " +
# #       "and 'time' - CBOW : ",
# #       clf.wv.similarity('love', 'time'))

# # w1 = "love"
# # print(clf.wv.most_similar(positive=w1))

# def get_vector(text):
#     tokenized_text = nltk.word_tokenize(text.lower())
#     vector = np.mean([clf.wv[word] for word in tokenized_text if word in clf.wv], axis=0)
#     return vector if len(vector) > 0 else np.zeros(clf.vector_size)

# # Build vectors for each tweet
# tweet_vectors = [get_vector(tweet) for tweet in data if any(word in clf.wv for word in nltk.word_tokenize(tweet.lower()))]

# # Calculate the average vector for all tweets
# average_vector = np.mean(tweet_vectors)
# print("\nAverage vectore=", average_vector)

# # user_input = input("Enter tweet: ")
# user_input = "the sun is shining brighly"
# user_vector = get_vector(user_input)
# print("\nUser vector=", user_vector)

# cos_sim = np.dot(user_vector, average_vector) / (np.linalg.norm(average_vector))
# print("\ncos sim=", cos_sim)


# # Combine all vectors for visualization
# all_vectors = np.vstack([tweet_vectors, user_vector])

# # Use t-SNE for dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# low_dimension = tsne.fit_transform(all_vectors)

# # Plot the results
# plt.scatter(low_dimension[:-1, 0], low_dimension[:-1, 1], label='Tweet Vectors')
# plt.scatter(low_dimension[-1, 0], low_dimension[-1, 1], label='User Input Vector', marker='X', color='red')
# # plt.scatter(low_dimension[-2, 0], low_dimension[-2, 1], label='Average Tweet Vector', marker='D', color='green')

# # Annotate points
# for i in range(len(data)-1):
#     plt.annotate(str(i), (low_dimension[i, 0], low_dimension[i, 1]))


# plt.legend()
# plt.show()
 