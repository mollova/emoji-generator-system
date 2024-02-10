# Results

## Dateset with 5 emojis

### Overall 63 119 tweets - 58 119 train and 5 000 test

|   Vectorizer  |  Classifier    |   Accuracy    |
| :-----------: | :-----------:  | :-----------: |
|  Bag of Words |  Naïve Bayes   |  47,82%       |
|  TF-IDF       |  Naïve Bayes   |  46,24%       |
|  Bag of Words |  kNN (k=5)     |  31,56%       |
|  TF-IDF       |  kNN (k=2)     |  30,68%       |
|  Bag of Words |  Random Forest |  27,72%       |
|  TF-IDF       |  Random Forest |  29,00%       |


### Overall 100 000 tweets - 95 000 train and 5 000 test

|   Vectorizer  |  Classifier    |   Accuracy    |
| :-----------: | :-----------:  | :-----------: |
|  Bag of Words |  Naïve Bayes   |  47,08%       |
|  TF-IDF       |  Naïve Bayes   |  47,02%       |
|  Bag of Words |  kNN (k=5)     |  31,42%       |
|  TF-IDF       |  kNN (k=3)     |  28,26%       |

## Dataset with 10 emojis

### Overall 121 813 tweets - 116 813 train and 5 000 test

|   Vectorizer  |  Classifier    |   Accuracy    |
| :-----------: | :-----------:  | :-----------: |
|  Bag of Words |  Naïve Bayes   |  35,04%       |
|  TF-IDF       |  Naïve Bayes   |  34,84%       |


### Overall 200 000 tweets - 195 000 train and 5 000 test

|   Vectorizer  |  Classifier    |   Accuracy    |
| :-----------: | :-----------:  | :-----------: |
|  Bag of Words |  Naïve Bayes   |  35,12%       |
|  TF-IDF       |  Naïve Bayes   |  35,34%       |
