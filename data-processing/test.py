from gensim.utils import lemmatize
 
sentence = "the bats saw the cats with best stripes hanging upside down by their feet"
 
lemmatized_sentence = [word.decode('utf-8').split('.')[0] for word in lemmatize(sentence)]
 
print(lemmatized_sentence)