# Results

## Dateset with 5 emojis

### Overall 63 119 tweets - 58 119 train and 5 000 test

|   Vectorizer  |  Classifier    |   Accuracy    |  Precision   |   Recall   |  F1-score   |    😭    |    😂     |   😤     |    🥹    |    😍    |
| :-----------: | :-----------:  | :-----------: | :----------: | :--------: | :---------: | :------: | :------: | :------: | :------: | :------: |
|  Bag of Words |  Naïve Bayes   |  47,82%       |  48,02%      |  45,78%    |  46,40%     |  49,33%  |  48,77%  |  40,49%  |  46,43%  |  46,96%  |
|  TF-IDF       |  Naïve Bayes   |  46,24%       |  48,59%      |  44,54%    |  45,27%     |  49,80%  |  49,87%  |  37,76%  |  44,07%  |  44,84%  |
|  Bag of Words |  kNN (k=5)     |  31,56%       |  29,89%      |  30,44%    |  28,97%     |  36,49%  |  30,47%  |  16,06%  |  26,15%  |  35,67%  |
|  TF-IDF       |  kNN (k=2)     |  30,68%       |  35,76%      |  25,75%    |  21,25%     |  27,83%  |  22,73%  |   8,91%  |  20,31%  |  26,46%  |
|  Bag of Words |  Random Forest |  27,72%       |  19,45%      |  23,15%    |  14,05%     |  41,84%  |  28,41%  |   0,00%  |   2,12%  |   1,08%  |
|  TF-IDF       |  Random Forest |  29,00%       |  19.46%      |  23.04%    |  13.90%     |  41.78%  |  27.74%  |   0,00%  |   0,00%  |    0,00% |


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
