import nltk
import random
from nltk.corpus import movie_reviews

documents = []

for category in movie_reviews.categories():
    # print(category)
    for file_id in movie_reviews.fileids(category):
        # print(file_id)
        # print(list(movie_reviews.words(file_id)))
        documents.append((list(movie_reviews.words(file_id)),category))

# print(documents[0])

random.shuffle(documents)

# print(documents[0])

all_words = []

for word in movie_reviews.words():
    all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)

print(all_words.most_common(15))

print(all_words["stupid"])






