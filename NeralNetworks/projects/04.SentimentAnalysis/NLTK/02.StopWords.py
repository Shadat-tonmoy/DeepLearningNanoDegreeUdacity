from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''

Stop words are basically words like a, the, and, for etc

For natural language processing these types of words should and has to be eliminated from sentences

NLTK helps greatly in removing these types of words from sentences

'''

example_sentence = "Welcome to a Natural Language Processing tutorial series, using the Natural Language Toolkit, or NLTK, module with Python."

stop_words = set(stopwords.words("english"))

# filtered_sentence = []

words = word_tokenize(example_sentence)

# for word in words :
#     if word not in stop_words:
#         filtered_sentence.append(word)

filtered_sentence = [word for word in words if word not in stop_words]

print(filtered_sentence)
