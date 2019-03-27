from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''

Stemming is the process of cut off sufix like ing, ed, s, es etc from words and get back the root word

For example, Write, Writing, Written, Writes will be stemmed to the root same word Write

NLTK use PorterStemmer to perform this stemming operation

'''

porter_stemmer = PorterStemmer()

example_words = ["Write", "Writing", "Written", "Writes", "Writer"]

for word in example_words:
    print(porter_stemmer.stem(word))

example_sentence = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once"

stop_words = set(stopwords.words())

words = word_tokenize(example_sentence)

print("Original Words\n",words)

filtered_words = [word for word in words if word not in stop_words]

print("Filtered Words\n",filtered_words)

stemmed_words = []
for word in filtered_words:
    stemmed_words.append(porter_stemmer.stem(word))

print("Stemmed Words\n",stemmed_words)



