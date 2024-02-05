import csv
from gensim.utils import simple_preprocess
from gensim import corpora

train_dataset = open('datasets/data/train_data_five_emojis.csv')
train_data_reader = csv.reader(train_dataset)

def tokenize_data():
    tokenized = []
    for row in train_data_reader:
        tokenized.append(simple_preprocess(str(row), deacc=True))

    dictionary = corpora.Dictionary(tokenized)
    dictionary.save('datasets/in-progress-data/dictionary.dict')
