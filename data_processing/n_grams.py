import pandas as pd
import matplotlib.pyplot as plt
from processing_data import create_dataframe

from collections import defaultdict

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

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

    plt.figure(1, figsize=(16, 8))
    plt.bar(pd1, pd2, color=color, width=0.4)
    plt.xlabel(xlabel=xlabel)
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )
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
