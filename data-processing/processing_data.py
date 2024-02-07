import preprocess_data
import csv
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train_dataset_name = 'datasets/data/train_data_five_emojis.csv'

train_data_dictionary_filepath = 'datasets/in-progress-data/dictionary.dict'
test_data_dictionary_filepath = 'datasets/in-progress-data/test-dictionary.dict'

def emoji_to_integer(df: pd.DataFrame):
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


def create_dataframe(dataset_filename: str, dictionary_path: str) -> pd.DataFrame:
    train_dataset = open(dataset_filename)
    train_data_reader = csv.reader(train_dataset)

    processed_data, emojis = preprocess_data.preprocessing_data(train_data_reader, dictionary_path)
    df = pd.DataFrame(list(zip(processed_data, emojis)), columns=['tweets', 'emojis'])

    # transform the emojis from string to integer
    emoji_to_integer(df)
    return df

def bag_of_words(df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    doc_term_matrix = vectorizer.fit_transform(df.tweets)
    vocab = vectorizer.get_feature_names_out()
    return pd.DataFrame(doc_term_matrix.todense(), columns=vocab)


def train_model_NB(vectorizer) -> MultinomialNB:
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    doc_term_df = bag_of_words(df=df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    clf = MultinomialNB(alpha=1, fit_prior=False)
    clf.fit(doc_term_df, target_values)
    return clf

def save_trained_model(model: MultinomialNB, model_filepath: str, vectorizer, vectorizer_filepath: str):
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


def test_bag_of_words(df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    doc_term_matrix = vectorizer.transform(df.tweets)
    vocab = vectorizer.get_feature_names_out()
    return pd.DataFrame(doc_term_matrix.todense(), columns=vocab)


def calculate_accuracy(clf: MultinomialNB, vectorizer, dataset_filename: str):
    df = create_dataframe(dataset_filename, test_data_dictionary_filepath) 
    doc_term_df = test_bag_of_words(df, vectorizer=vectorizer) 
    target_values = df.emojis.astype(int)   

    predicted_emojis = clf.predict(doc_term_df)
    print(predicted_emojis[:10])
    print(target_values[:10])
    print("Accuracy: ", accuracy_score(target_values, predicted_emojis))
