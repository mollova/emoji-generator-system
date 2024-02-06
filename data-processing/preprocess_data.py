import csv
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords
from typing import Any

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')


def preprocessing_data(train_data_reader: Any):
    tokenized = []
    emojis = []
    for row in train_data_reader:
        emojis.append(row[-1][-1])
        tokenized.append(simple_preprocess(remove_stopwords(str(row))))

    dictionary = corpora.Dictionary(tokenized)
    dictionary.save('datasets/in-progress-data/dictionary.dict')

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
        processed_data, emojis = preprocessing_data(train_data_reader=train_data_reader)
        for i in range(1,len(processed_data)):
            file.writelines(processed_data[i] + ' ' + emojis[i] + '\n')


save_lemmatized_data()