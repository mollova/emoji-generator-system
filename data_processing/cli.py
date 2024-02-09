import cmd
import classifiers
from simple_term_menu import TerminalMenu

BAG_OF_WORDS = "Bag of Words"
TF_IDF = "TF-IDF"

NAIVE_BAYES = "Naïve Bayes"
KNN = "kNN"
SVM = "SVM"
# RANDOM_FOREST = "Random Forest"

args_to_functions = {
    (BAG_OF_WORDS, NAIVE_BAYES): classifiers.test_bow_and_nb_cli,
    (BAG_OF_WORDS, KNN): classifiers.test_bow_and_knn_cli,
    (BAG_OF_WORDS, SVM): classifiers.test_bow_and_svm_cli,

    (TF_IDF, NAIVE_BAYES): classifiers.test_tfidf_and_nb,
    (TF_IDF, KNN): classifiers.test_tfidf_and_knn_cli,
    (TF_IDF, SVM): classifiers.test_tfidf_and_svm_cli
}

class EmojiCLI(cmd.Cmd):
    prompt = '>> '
    intro = 'Welcome to Emoji Genarator System.'

    def preloop(self):
        print("Choose the data processing type:")
        data_proccessing_options = [BAG_OF_WORDS, TF_IDF]
        terminal_menu = TerminalMenu(data_proccessing_options)
        choice_index = terminal_menu.show()
        print(f"You have selected {data_proccessing_options[choice_index]}!")
        self.vectorization_type = data_proccessing_options[choice_index]

        print("Choose the classifier type:")
        classifier_options = [NAIVE_BAYES, KNN, SVM]
        terminal_menu = TerminalMenu(classifier_options)
        choice_index = terminal_menu.show()
        print(f"You have selected {classifier_options[choice_index]}!")
        self.classifier_type = classifier_options[choice_index]

    def default(self, text: str) -> None:
        result = args_to_functions[(self.vectorization_type, self.classifier_type)](text)
        print("Emoji: ", result)
    
    def do_quit(self, line: str) -> None:
        """Exit the CLI."""
        return True

if __name__ == '__main__':
    EmojiCLI().cmdloop()
    