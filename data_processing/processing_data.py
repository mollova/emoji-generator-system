from sklearn.base import ClassifierMixin
import preprocess_data
import csv
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from gensim.utils import simple_preprocess
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


import nltk
nltk.download('punkt')
nltk.download('wordnet')

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

train_dataset_name = 'datasets/data/train_data_five_emojis.csv'

train_data_dictionary_filepath = 'datasets/in-progress-data/dictionary.dict'
test_data_dictionary_filepath = 'datasets/in-progress-data/test-dictionary.dict'

def emoji_to_integer(df: pd.DataFrame) -> None:
    df.loc[df.emojis == '😭', 'emojis' ] = 0
    df.loc[df.emojis == '😂', 'emojis' ] = 1
    df.loc[df.emojis == '😤', 'emojis' ] = 2
    df.loc[df.emojis == '🥹', 'emojis' ] = 3
    df.loc[df.emojis == '😍', 'emojis' ] = 4
    # df.loc[df.emojis == '🤡', 'emojis' ] = 5
    # df.loc[df.emojis == '🥵', 'emojis' ] = 6
    # df.loc[df.emojis == '💀', 'emojis' ] = 7
    # df.loc[df.emojis == '🤔', 'emojis' ] = 8
    # df.loc[df.emojis == '😉', 'emojis' ] = 9


def integer_to_emoji(number: int) -> str:
    if number == 0:
        return '😭'
    elif number == 1:
        return '😂'
    elif number == 2:
        return '😤'
    elif number == 3:
        return '🥹'
    elif number == 4:
        return '😍'
    # elif number == 5:
    #     return '🤡'
    # elif number == 6:
    #     return '🥵'
    # elif number == 7:
    #     return '💀'
    # elif number == 8:
    #     return '🤔'
    # elif number == 9:
    #     return '😉'
    else:
        return 'Invalid number'

def create_dataframe(dataset_filename: str, dictionary_path: str, should_parse: bool = True) -> pd.DataFrame:
    train_dataset = open(dataset_filename)
    train_data_reader = csv.reader(train_dataset)

    processed_data, emojis = preprocess_data.preprocessing_data(train_data_reader, dictionary_path)
    df = pd.DataFrame(list(zip(processed_data, emojis)), columns=['tweets', 'emojis'])

    # transform the emojis from string to integer
    if should_parse:
        emoji_to_integer(df)

    return df

def create_dataframe_cli(text: str) -> pd.DataFrame:
    words=[word for word in str(text).split(" ") if word not in set(stopwords.words('english'))]
    tokenized = simple_preprocess(' '.join(words))
    df = pd.DataFrame(tokenized, columns=['tweets'])

    return df

def vectorize(df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    doc_term_matrix = vectorizer.fit_transform(df.tweets)
    vocab = vectorizer.get_feature_names_out()

    return pd.DataFrame(doc_term_matrix.todense(), columns=vocab)

def test_data_vectorize(df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    doc_term_matrix = vectorizer.transform(df.tweets)
    vocab = vectorizer.get_feature_names_out()
    return pd.DataFrame(doc_term_matrix.todense(), columns=vocab)

def train_model_NB(vectorizer) -> MultinomialNB:
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    doc_term_df = vectorize(df=df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    clf = MultinomialNB(alpha=1, fit_prior=False)
    clf.fit(doc_term_df, target_values)

    return clf


def train_model_SVM(vectorizer) -> svm:
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    doc_term_df = vectorize(df=df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    clf = svm.SVC(kernel='linear')
    clf.fit(doc_term_df, target_values)

    return clf


def train_model_KNN(vectorizer):
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    doc_term_df = vectorize(df=df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(doc_term_df, target_values)

    return knn


def train_model_random_forest(vectorizer):
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    doc_term_df = vectorize(df=df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    random_forest = RandomForestClassifier()
    random_forest.fit(doc_term_df, target_values)

    return random_forest


def save_trained_model(model: ClassifierMixin, model_filepath: str, vectorizer, vectorizer_filepath: str):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

    with open(vectorizer_filepath, 'wb') as file:
        pickle.dump(vectorizer, file)


def load_trained_model(model_filepath: str, vectorizer_filepath: str):
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)

    with open(vectorizer_filepath, 'rb') as file:
        vectorizer = pickle.load(file)

    return model, vectorizer

def calculate_accuracy(clf: ClassifierMixin, vectorizer, dataset_filename: str):
    df = create_dataframe(dataset_filename, test_data_dictionary_filepath)
    doc_term_df = test_data_vectorize(df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    predicted_emojis = clf.predict(doc_term_df)
    print("Accuracy: ", accuracy_score(target_values, predicted_emojis))


def predict_emoji_cli(clf: ClassifierMixin, vectorizer, text: str):
    df = create_dataframe_cli(text, test_data_dictionary_filepath)
    doc_term_df = test_data_vectorize(df, vectorizer=vectorizer)
    predicted_emoji = clf.predict(doc_term_df)
    return integer_to_emoji(predicted_emoji[0])
