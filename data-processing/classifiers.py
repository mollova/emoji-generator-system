import processing_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

test_dataset_name = 'datasets/data/test_data_five_emojis.csv'

bow_nb_model_filepath = 'classifier_models/bow_nb.pkl'
count_vectorizer_filepath = 'classifier_models/count_vectorizer.pkl'

tfidf_nb_model_filepath = 'classifier_models/tfidf_nb.pkl'
tfidf_vectorizer_filepath = 'classifier_models/tfidf_vectorizer.pkl'

def train_bow_and_nb():
    vectorizer = CountVectorizer()
    clf = processing_data.train_model_NB(vectorizer)
    processing_data.save_trained_model(clf, bow_nb_model_filepath, vectorizer, count_vectorizer_filepath)

def test_bow_and_nb():
    model, vectorizer = processing_data.load_trained_model(bow_nb_model_filepath, count_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def train_tfidf_and_nb():
    vectorizer = TfidfVectorizer()
    clf = processing_data.train_model_NB(vectorizer)
    processing_data.save_trained_model(clf, tfidf_nb_model_filepath, vectorizer, tfidf_vectorizer_filepath)

def test_tfidf_and_nb():
    model, vectorizer = processing_data.load_trained_model(tfidf_nb_model_filepath, tfidf_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def bow_and_nb():
    train_bow_and_nb()
    test_bow_and_nb()

def tfidf_and_nb():
    train_tfidf_and_nb()
    test_tfidf_and_nb()

# bow_and_nb()
# test_bow_and_nb()

tfidf_and_nb()
