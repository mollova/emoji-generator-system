import csv
from gensim.utils import simple_preprocess
from gensim import corpora
from typing import Any

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

train_data_dictionary_filepath = 'datasets/in-progress-data/dictionary.dict'

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def preprocessing_data(train_data_reader: Any, dictionary_path: str):
    tokenized = []
    emojis = []
    for row in train_data_reader:
        emojis.append(row[-1][-1])
        words=[word for word in str(row).split(" ") if word not in set(stopwords.words('english'))]
        tokenized.append(simple_preprocess(' '.join(words)))

    dictionary = corpora.Dictionary(tokenized)
    dictionary.save(dictionary_path)

    lemmatized = []
    lemmatizer = WordNetLemmatizer()
    for tokenized_sentence in tokenized:
        lem_words = [lemmatizer.lemmatize(word) for word in tokenized_sentence]
        lemmatized.append(' '.join(lem_words))

    return lemmatized, emojis


def save_lemmatized_data() -> None:
    train_dataset = open('datasets/data/train_data_five_emojis.csv', 'r')
    train_data_reader = csv.reader(train_dataset)

    with open('datasets/in-progress-data/processed_train_data_five_emojis.csv', 'w') as file:
        processed_data, emojis = preprocessing_data(train_data_reader=train_data_reader, dictionary_path=train_data_dictionary_filepath)
        for i in range(1,len(processed_data)):
            file.writelines(processed_data[i] + ' ' + emojis[i] + '\n')


# save_lemmatized_data()