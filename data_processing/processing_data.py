from matplotlib.font_manager import FontProperties
from sklearn.base import ClassifierMixin
import preprocess_data
import csv
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from gensim.utils import simple_preprocess
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import gensim
import numpy as np
import matplotlib.pyplot as plt




import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

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

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(doc_term_df, target_values)

    return knn


def train_model_random_forest(vectorizer):
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    doc_term_df = vectorize(df=df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    random_forest = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)
    random_forest.fit(doc_term_df, target_values)

    return random_forest


def train_model_word2vec():
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    nlp = [nltk.word_tokenize(i) for i in df['tweets']]
    model = gensim.models.Word2Vec(nlp, min_count=1, vector_size=100, window=5)

    return model

def train_nb_word2vec():
    # Step 3: Create Word2Vec vectors
    def get_vector(text):
        tokenized_text = text.lower().split(' ')
        vector = np.mean([model.wv[word] for word in tokenized_text if word in model.wv], axis=0)
        return vector if vector is not None else np.zeros(model.vector_size)

    model = train_model_word2vec()
    df = create_dataframe(train_dataset_name, train_data_dictionary_filepath)
    # doc_term_df = df['tweets']
    doc_term_df = [get_vector(tweet) for tweet in df['tweets']]
    variable2 = np.asarray(doc_term_df, dtype="object")
    # doc_term_df = np.vstack([get_vector(tweet) for tweet in df['tweets']])
    target_values = df.emojis.astype(int)

    nb = GaussianNB()
    print("doc term df: ", doc_term_df)
    print("\n target: ", target_values)
    nb.fit(variable2, list(target_values))

    return nb


def save_trained_model(model: ClassifierMixin, model_filepath: str, vectorizer=None, vectorizer_filepath: str = None):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

    if vectorizer:
        with open(vectorizer_filepath, 'wb') as file:
            pickle.dump(vectorizer, file)


def load_trained_model(model_filepath: str, vectorizer_filepath: str = None):
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)

    vectorizer = None
    if vectorizer_filepath:
        with open(vectorizer_filepath, 'rb') as file:
            vectorizer = pickle.load(file)

    return model, vectorizer

def calculate_accuracy(clf: ClassifierMixin, vectorizer, dataset_filename: str):
    df = create_dataframe(dataset_filename, test_data_dictionary_filepath)
    doc_term_df = test_data_vectorize(df, vectorizer=vectorizer)
    target_values = df.emojis.astype(int)

    predicted_emojis = clf.predict(doc_term_df)
    print("Accuracy: ", accuracy_score(target_values, predicted_emojis))

    labels = [str(x) for x in range(5)]

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        target_values, predicted_emojis, labels=labels, average="macro")
    p_list, r_list, f1_list, freq_list = precision_recall_fscore_support(
        target_values, predicted_emojis, labels=labels, average=None)

    print("\nprecision: ", p_macro)
    print("\nrecall: ", r_macro)
    print("\nf1: ", f1_macro)
    for i in range(5):
        print(f"emoji {i}: ", integer_to_emoji(i), " emoji's acc: ", f1_list[i])

    save_confussion_matrix(target_values, predicted_emojis)


def save_confussion_matrix(target_values, predicted_values):
    confusion_matrix = metrics.confusion_matrix(target_values, predicted_values)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                        display_labels=[integer_to_emoji(1), integer_to_emoji(2), '😂', '😭', '😍'])

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the plot
    cm_display.plot(ax=ax)
    plt.show()

    # Save the plot
    fig.savefig("confusion_matrix.png")


def predict_emoji_cli(clf: ClassifierMixin, vectorizer, text: str):
    df = create_dataframe_cli(text)
    doc_term_df = test_data_vectorize(df, vectorizer=vectorizer)
    predicted_emoji = clf.predict(doc_term_df)
    return integer_to_emoji(predicted_emoji[0])