# import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# from processing_data import create_dataframe

# from collections import defaultdict

# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# train_dataset_name = 'datasets/data/train_data_five_emojis.csv'
# train_data_dictionary_filepath = 'datasets/in-progress-data/dictionary.dict'


# def generate_N_grams(text,ngram=1):
#     words = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
#     temp = zip(*[words[i:] for i in range(0,ngram)])
#     ans = [' '.join(ngram) for ngram in temp]

#     return ans


# def plot_n_gram(dataframe: pd.DataFrame,
#                 color: str,
#                 xlabel: str,
#                 ylabel:str,
#                 title: str,
#                 save_name: str) -> None:

#     pd1=dataframe[0][:10]
#     pd2=dataframe[1][:10]

#     plt.figure(1,figsize=(16,4))
#     plt.bar(pd1,pd2, color=color,
#             width = 0.4)
#     plt.xlabel(xlabel=xlabel)
#     plt.ylabel(ylabel=ylabel)
#     plt.title(label=title)
#     plt.savefig(save_name)
#     plt.show()

# N_GRAM_SIZE=2
# if N_GRAM_SIZE == 1:
#     ngram_word = "unigram"
# elif N_GRAM_SIZE == 2:
#     ngram_word = "bigram"
# elif N_GRAM_SIZE == 3:
#     ngram_word = "trigram"

# laughing_emoji = defaultdict(int)
# holding_tears_emoji = defaultdict(int)
# nose_steam_emoji = defaultdict(int)
# crying_emoji = defaultdict(int)
# heart_eyes_emoji = defaultdict(int)

# df = create_dataframe(dataset_filename=train_dataset_name, dictionary_path=train_data_dictionary_filepath, should_parse=False)

# for text in df[df.emojis=="😂"].tweets:
#     for word in generate_N_grams(text,N_GRAM_SIZE):
#         laughing_emoji[word] += 1

# for text in df[df.emojis=="🥹"].tweets:
#     for word in generate_N_grams(text,N_GRAM_SIZE):
#         holding_tears_emoji[word] += 1

# for text in df[df.emojis=="😤"].tweets:
#     for word in generate_N_grams(text,N_GRAM_SIZE):
#         nose_steam_emoji[word] += 1

# for text in df[df.emojis=="😭"].tweets:
#     for word in generate_N_grams(text,N_GRAM_SIZE):
#         crying_emoji[word] += 1

# for text in df[df.emojis=="😍"].tweets:
#     for word in generate_N_grams(text,N_GRAM_SIZE):
#         heart_eyes_emoji[word] += 1

# df_laughing_emoji = pd.DataFrame(sorted(laughing_emoji.items(),key=lambda x:x[1],reverse=True))
# df_holding_tears_emoji = pd.DataFrame(sorted(holding_tears_emoji.items(),key=lambda x:x[1],reverse=True))
# df_nose_steam_emoji = pd.DataFrame(sorted(nose_steam_emoji.items(),key=lambda x:x[1],reverse=True))
# df_crying_emoji = pd.DataFrame(sorted(crying_emoji.items(),key=lambda x:x[1],reverse=True))
# df_heart_eyes_emoji = pd.DataFrame(sorted(heart_eyes_emoji.items(),key=lambda x:x[1],reverse=True))

# plot_n_gram(dataframe=df_laughing_emoji,
#             color='green',
#             xlabel=f"Most common {ngram_word} with laughing emoji - 😂",
#             ylabel="Count",
#             title=f"Ten most popular {ngram_word} with 😂",
#             save_name=f"laughing_{ngram_word}.png")

# plot_n_gram(dataframe=df_holding_tears_emoji,
#             color='green',
#             xlabel=f"Most common {ngram_word} with holding tears emoji - 🥹",
#             ylabel="Count",
#             title=f"Ten most popular {ngram_word} with 🥹",
#             save_name=f"holding_tears_{ngram_word}.png")

# plot_n_gram(dataframe=df_nose_steam_emoji,
#             color='green',
#             xlabel=f"Most common {ngram_word} with steam from nose emoji - 😤",
#             ylabel="Count",
#             title=f"Ten most popular {ngram_word} with 😤",
#             save_name=f"steam_from_nose_{ngram_word}.png")

# plot_n_gram(dataframe=df_crying_emoji,
#             color='green',
#             xlabel=f"Most common {ngram_word} with crying emoji - 😭",
#             ylabel="Count",
#             title=f"Ten most popular {ngram_word} with 😭",
#             save_name=f"crying_{ngram_word}.png")

# plot_n_gram(dataframe=df_heart_eyes_emoji,
#             color='green',
#             xlabel=f"Most common {ngram_word} with heart eyed emoji - 😍",
#             ylabel="Count",
#             title=f"Ten most popular {ngram_word} with 😍",
#             save_name=f"heart_eyes_{ngram_word}.png")


import csv
import pandas as pd
import matplotlib.pyplot as plt
from processing_data import create_dataframe

from collections import defaultdict

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def generate_N_grams(text, ngram=1):
    words = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]

    return ans

def count_emoji_ngrams(df, emoji, ngram_size=1):
    emoji_counts = defaultdict(int)

    for text in df[df.emojis == emoji].tweets:
        for word in generate_N_grams(text, ngram_size):
            emoji_counts[word] += 1

    return pd.DataFrame(sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True))

def plot_n_gram(dataframe, color, xlabel, ylabel, title, save_name):
    pd1 = dataframe[0][:10]
    pd2 = dataframe[1][:10]

    plt.figure(1, figsize=(16, 4))
    plt.bar(pd1, pd2, color=color, width=0.4)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(label=title)
    plt.savefig(save_name)
    plt.show()

def process_emoji_data(df, emoji, ngram_size=1, color='green'):
    df_emoji = count_emoji_ngrams(df, emoji, ngram_size)

    ngram_word = f"{ngram_size}-gram"
    emoji_label = f"Most common {ngram_word} with {emoji} emoji"

    plot_n_gram(df_emoji, color, xlabel=emoji_label, ylabel="Count",
                title=f"Ten most popular {ngram_word} with {emoji}", save_name=f"{emoji.lower()}_{ngram_word}.png")

train_dataset_name = 'datasets/data/train_data_five_emojis.csv'
train_data_dictionary_filepath = 'datasets/in-progress-data/dictionary.dict'

df = create_dataframe(dataset_filename=train_dataset_name, dictionary_path=train_data_dictionary_filepath, should_parse=False)

for i in [1,2,3]:
    emojis = ["😂", "🥹", "😤", "😭", "😍"]
    for emoji in emojis:
        process_emoji_data(df, emoji, ngram_size=i, color='green')
