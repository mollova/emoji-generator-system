import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import preprocess_data

train_dataset_name = 'datasets/in-progress-data/processed_train_data_five_emojis.csv'
test_dataset_name = 'datasets/data/test_data_five_emojis.csv'

def emoji_to_integer(df: pd.DataFrame):
    df.loc[df.emojis == '😭', 'emojis' ] = 0
    df.loc[df.emojis == '😂', 'emojis' ] = 1
    df.loc[df.emojis == '😤', 'emojis' ] = 2
    df.loc[df.emojis == '🥹', 'emojis' ] = 3
    df.loc[df.emojis == '😍', 'emojis' ] = 4

def create_dataframe(dataset_filename: str) -> pd.DataFrame:
    train_dataset = open(dataset_filename)
    train_data_reader = csv.reader(train_dataset)

    # tweets = []
    # emojis = []
    # for row in train_data_reader:
    #     tweet = row[-1][:-1]
    #     emoji = row[-1][-1]
    #     tweets.append(tweet)
    #     emojis.append(emoji)

    processed_data, emojis = preprocess_data.preprocessing_data(train_data_reader)
    df = pd.DataFrame(list(zip(processed_data, emojis)), columns=['tweets', 'emojis'])

    # transform the emojis from string to integer
    emoji_to_integer(df)
    return df
    
def bag_of_words(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(df.tweets)

    vocab = vectorizer.get_feature_names_out()
    return pd.DataFrame(doc_term_matrix.todense(), columns=vocab)

def train_model_NB() -> MultinomialNB:
    df = create_dataframe(train_dataset_name)
    doc_term_df = bag_of_words(df)
    target_values = df.emojis.astype(int)

    clf = MultinomialNB()
    clf.fit(doc_term_df, target_values)
    return clf

def calculate_accuracy(clf: MultinomialNB, dataset_filename: str):
    df = create_dataframe(dataset_filename) 
    doc_term_df = bag_of_words(df) 
    target_values = df.emojis.astype(int)   

    predicted_emojis = clf.predict(doc_term_df)
    print(predicted_emojis)
    print("Accuracy: ", accuracy_score(target_values, predicted_emojis))

clf = train_model_NB()
calculate_accuracy(clf, test_dataset_name)

