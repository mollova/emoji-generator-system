import processing_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

test_dataset_name = 'datasets/data/test_data_five_emojis.csv'

bow_nb_model_filepath = 'classifier_models/demo/bow_nb.pkl'
bow_knn_model_filepath = 'classifier_models/demo/bow_knn.pkl'
bow_svm_model_filepath = 'classifier_models/demo/bow_svm.pkl'
bow_random_forest_model_filepath = 'classifier_models/demo/bow_random_forest.pkl'
count_vectorizer_filepath = 'classifier_models/demo/count_vectorizer.pkl'

tfidf_nb_model_filepath = 'classifier_models/demo/tfidf_nb.pkl'
tfidf_knn_model_filepath = 'classifier_models/demo/tfidf_knn.pkl'
tfidf_svm_model_filepath = 'classifier_models/demo/tfidf_svm.pkl'
tfidf_random_forest_model_filepath = 'classifier_models/demo/tfidf_random_forest.pkl'
tfidf_vectorizer_filepath = 'classifier_models/demo/tfidf_vectorizer.pkl'

def train_bow_and_nb():
    vectorizer = CountVectorizer()
    clf = processing_data.train_model_NB(vectorizer)
    processing_data.save_trained_model(clf, bow_nb_model_filepath, vectorizer, count_vectorizer_filepath)

def test_bow_and_nb():
    model, vectorizer = processing_data.load_trained_model(bow_nb_model_filepath, count_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_bow_and_nb_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(bow_nb_model_filepath, count_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def train_tfidf_and_nb():
    vectorizer = TfidfVectorizer()
    clf = processing_data.train_model_NB(vectorizer)
    processing_data.save_trained_model(clf, tfidf_nb_model_filepath, vectorizer, tfidf_vectorizer_filepath)

def test_tfidf_and_nb():
    model, vectorizer = processing_data.load_trained_model(tfidf_nb_model_filepath, tfidf_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_tfidf_and_nb_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(tfidf_nb_model_filepath, tfidf_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def train_bow_and_knn():
    vectorizer = CountVectorizer()
    clf = processing_data.train_model_KNN(vectorizer)
    processing_data.save_trained_model(clf, bow_knn_model_filepath, vectorizer, count_vectorizer_filepath)

def test_bow_and_knn():
    model, vectorizer = processing_data.load_trained_model(bow_knn_model_filepath, count_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_bow_and_knn_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(bow_knn_model_filepath, count_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def train_tfidf_and_knn():
    vectorizer = TfidfVectorizer()
    clf = processing_data.train_model_KNN(vectorizer)
    processing_data.save_trained_model(clf, tfidf_knn_model_filepath, vectorizer, tfidf_vectorizer_filepath)

def test_tfidf_and_knn():
    model, vectorizer = processing_data.load_trained_model(tfidf_knn_model_filepath, tfidf_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_tfidf_and_knn_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(tfidf_knn_model_filepath, tfidf_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def train_bow_and_random_forest():
    vectorizer = CountVectorizer()
    clf = processing_data.train_model_random_forest(vectorizer)
    processing_data.save_trained_model(clf, bow_random_forest_model_filepath, vectorizer, count_vectorizer_filepath)

def test_bow_and_random_forest():
    model, vectorizer = processing_data.load_trained_model(bow_random_forest_model_filepath, count_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_bow_and_random_forest_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(bow_random_forest_model_filepath, count_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)

def train_tfidf_and_random_forest():
    vectorizer = TfidfVectorizer()
    clf = processing_data.train_model_random_forest(vectorizer)
    processing_data.save_trained_model(clf, tfidf_random_forest_model_filepath, vectorizer, tfidf_vectorizer_filepath)

def test_tfidf_and_random_forest():
    model, vectorizer = processing_data.load_trained_model(tfidf_random_forest_model_filepath, tfidf_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_tfidf_and_random_forest_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(tfidf_random_forest_model_filepath, tfidf_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def train_bow_and_svm():
    vectorizer = CountVectorizer()
    clf = processing_data.train_model_SVM(vectorizer=vectorizer)
    processing_data.save_trained_model(clf, bow_svm_model_filepath, vectorizer, count_vectorizer_filepath)

def test_bow_and_svm():
    model, vectorizer = processing_data.load_trained_model(bow_svm_model_filepath, count_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_bow_and_svm_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(bow_svm_model_filepath, count_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def train_tfidf_and_svm():
    vectorizer = TfidfVectorizer()
    clf = processing_data.train_model_SVM(vectorizer=vectorizer)
    processing_data.save_trained_model(clf, tfidf_svm_model_filepath, vectorizer, tfidf_vectorizer_filepath)

def test_tfidf_and_svm():
    model, vectorizer = processing_data.load_trained_model(tfidf_svm_model_filepath, tfidf_vectorizer_filepath)
    processing_data.calculate_accuracy(model, vectorizer, test_dataset_name)

def test_tfidf_and_svm_cli(input: str):
    model, vectorizer = processing_data.load_trained_model(tfidf_svm_model_filepath, tfidf_vectorizer_filepath)

    return processing_data.predict_emoji_cli(model, vectorizer, input)


def bow_and_nb():
    print("Bag of Words + Naïve Bayes")
    train_bow_and_nb()
    test_bow_and_nb()

def tfidf_and_nb():
    print("TF-IDF + Naïve Bayes")
    train_tfidf_and_nb()
    test_tfidf_and_nb()

def bow_and_knn():
    print("Bag of Words + kNN")
    train_bow_and_knn()
    test_bow_and_knn()

def tfidf_knn():
    print("TF-IDF + kNN")
    train_tfidf_and_knn()
    test_tfidf_and_knn()

def bow_and_random_forest():
    print("Bag of Words + Random Forest")
    train_bow_and_svm()
    test_bow_and_svm()

def tfidf_and_random_forest():
    print("TF-IDF + Random Forest")
    train_tfidf_and_random_forest()
    test_tfidf_and_random_forest()

def bow_and_svm():
    print("Bag of Words + SVM")
    train_bow_and_svm()
    test_bow_and_svm()

def tfidf_and_svm():
    print("TF-IDF + SVM")
    train_tfidf_and_svm()
    test_tfidf_and_svm()


# bow_and_nb()
# print()

# tfidf_and_nb()
# print()

# bow_and_knn()
# print()

# tfidf_knn()
# print()

# bow_and_svm()
# print()

# tfidf_and_svm()
# print()

bow_and_random_forest()
print()

# tfidf_and_random_forest()
# print()