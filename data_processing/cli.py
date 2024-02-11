import cmd
import classifiers
import word2vec
from simple_term_menu import TerminalMenu

USE_CLASSIFICATION_MODEL = "Classification model"
USE_VECTOR_SIMILARITY = "Vector similarity"

MAX_SIMILARITY = "Return the closest tweet emoji"
AVERAGE_SIMILARITY = "Return emoji with best average score"

JACCARD_SIMILARITY = "Jaccard similarity"
COSINE_SIMILARITY = "Cosine similarity"

BAG_OF_WORDS = "Bag of Words"
TF_IDF = "TF-IDF"

NAIVE_BAYES = "Naïve Bayes"
KNN = "kNN"
# SVM = "SVM"
RANDOM_FOREST = "Random Forest"

NONE_MARKER = None

args_to_functions = {
    (USE_CLASSIFICATION_MODEL, BAG_OF_WORDS, NAIVE_BAYES, NONE_MARKER, NONE_MARKER): classifiers.test_bow_and_nb_cli,
    (USE_CLASSIFICATION_MODEL, BAG_OF_WORDS, KNN, NONE_MARKER, NONE_MARKER): classifiers.test_bow_and_knn_cli,
    # (BAG_OF_WORDS, SVM): classifiers.test_bow_and_svm_cli,
    (USE_CLASSIFICATION_MODEL, BAG_OF_WORDS, RANDOM_FOREST, NONE_MARKER, NONE_MARKER): classifiers.test_bow_and_random_forest_cli,

    (USE_CLASSIFICATION_MODEL, TF_IDF, NAIVE_BAYES, NONE_MARKER, NONE_MARKER): classifiers.test_tfidf_and_nb_cli,
    (USE_CLASSIFICATION_MODEL, TF_IDF, KNN, NONE_MARKER, NONE_MARKER): classifiers.test_tfidf_and_knn_cli,
    # (TF_IDF, SVM): classifiers.test_tfidf_and_svm_cli,
    (USE_CLASSIFICATION_MODEL, TF_IDF, RANDOM_FOREST, NONE_MARKER, NONE_MARKER): classifiers.test_tfidf_and_random_forest_cli,

    (USE_VECTOR_SIMILARITY, NONE_MARKER, NONE_MARKER, JACCARD_SIMILARITY, MAX_SIMILARITY): word2vec.suggest_emoji_max_jaccard_similarity,
    (USE_VECTOR_SIMILARITY, NONE_MARKER, NONE_MARKER, JACCARD_SIMILARITY, AVERAGE_SIMILARITY): word2vec.suggest_emoji_average_jaccard_similarity,
    (USE_VECTOR_SIMILARITY, NONE_MARKER, NONE_MARKER, COSINE_SIMILARITY, NONE_MARKER): word2vec.max_cosine_similarity
}

class EmojiCLI(cmd.Cmd):
    workflow_option = None
    vectorization_type = None
    classifier_type = None
    vector_similarity_option = None
    jaccard_option = None

    prompt = '>> '
    intro = 'Welcome to Emoji Genarator System.'


    def preloop(self):
        print("Choose program workflow:")
        workflow_options = [USE_CLASSIFICATION_MODEL, USE_VECTOR_SIMILARITY]
        terminal_menu = TerminalMenu(workflow_options)
        choice_index = terminal_menu.show()
        print(f"You have selected {workflow_options[choice_index]}")
        self.workflow_option = workflow_options[choice_index]
        if workflow_options[choice_index] == USE_CLASSIFICATION_MODEL:
            print("Choose the data processing type:")
            data_proccessing_options = [BAG_OF_WORDS, TF_IDF]
            terminal_menu = TerminalMenu(data_proccessing_options)
            choice_index = terminal_menu.show()
            print(f"You have selected {data_proccessing_options[choice_index]}!")
            self.vectorization_type = data_proccessing_options[choice_index]

            print("Choose the classifier type:")
            classifier_options = [NAIVE_BAYES, KNN, RANDOM_FOREST]
            terminal_menu = TerminalMenu(classifier_options)
            choice_index = terminal_menu.show()
            print(f"You have selected {classifier_options[choice_index]}!")
            self.classifier_type = classifier_options[choice_index]
        elif workflow_options[choice_index] == USE_VECTOR_SIMILARITY:
            print("Choose vector similarity type:")
            vector_similarity_options = [JACCARD_SIMILARITY, COSINE_SIMILARITY]
            terminal_menu = TerminalMenu(vector_similarity_options)
            choice_index = terminal_menu.show()
            print(f"You have selected {vector_similarity_options[choice_index]}")
            self.vector_similarity_option = vector_similarity_options[choice_index]
            if vector_similarity_options[choice_index] == JACCARD_SIMILARITY:
                print("Choose:")
                jaccard_options = [MAX_SIMILARITY, AVERAGE_SIMILARITY]
                terminal_menu = TerminalMenu(jaccard_options)
                choice_index = terminal_menu.show()
                print(f"You have selected {jaccard_options[choice_index]}!")
                self.jaccard_option = jaccard_options[choice_index]

        else:
            pass



    def default(self, text: str) -> None:
        result = args_to_functions[(self.workflow_option,
                                    self.vectorization_type,
                                    self.classifier_type,
                                    self.vector_similarity_option,
                                    self.jaccard_option
                                )](text)
        print("Suggested emoji: ", result)

    def do_quit(self, line: str) -> None:
        """Exit the CLI."""
        return True

if __name__ == '__main__':
    EmojiCLI().cmdloop()
