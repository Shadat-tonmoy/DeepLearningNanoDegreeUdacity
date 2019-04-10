from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize

'''

Corpora is basically corpus of Data. NLTK has a bunch of built in data set in its corpus package

'''

sample_text = movie_reviews.raw("neg/cv000_29416.txt")

sentence_tokens = sent_tokenize(sample_text)

print(sentence_tokens[1:10])
