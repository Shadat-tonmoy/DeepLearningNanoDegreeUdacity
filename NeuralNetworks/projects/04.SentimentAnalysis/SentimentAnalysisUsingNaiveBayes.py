import nltk
import random
from nltk.corpus import movie_reviews

documents = []

for category in movie_reviews.categories():
    # print(category)                                       # positive or negative category
    for file_id in movie_reviews.fileids(category):         # all the files with positive or negative review category
        # print(file_id)                                    # printing the file name
        # print(list(movie_reviews.words(file_id)))         # print all the words from a file of movie review
        documents.append((list(movie_reviews.words(file_id)),category))
                                                            # adding a movie review words and category (pos/neg) as a tuple

# print(documents[0])

random.shuffle(documents)               # randomly shuffling the documents for making variation in training data

# print(documents[0])

all_words = []                          # list of all words from all movie review

for word in movie_reviews.words():
    all_words.append(word.lower())      # appending the words in all word list

all_words = nltk.FreqDist(all_words)   # making a frequency distribution of all words from all reviews (how many times a word appears

# print(all_words.most_common(15))      # printing the most common 15 words

# print(all_words["stupid"])            # printing the frequency of the word 'stupid'


word_features = list(all_words.keys())[:3000]   # listing the top 3000 from most common words as features. We care about only the
                                                # word but not their frequency
print(word_features)


def find_features(document):                    # function to find features from a documents
    words = set(document)                       # extracting the unique words from a document
    # print("FeaturesWords")
    # print(words)
    features = {}                               # dictionary of features that indicates whether a word (as a feature) from the top
                                                # 3000 words is present in the word set of the documents
    for word_feature in word_features:
        features[word_feature] = (word_feature in words)    # if 'word' is present then features['word'] =  true and false otherwise

    return features

# print(find_features(list(movie_reviews.words('neg/cv000_29416.txt'))))    # testing the find feature function

feature_set = []        # feature set is a list of tuples where each tuple contains a dictionary with the top 3000 words as key
                        # and True or False as their appearance in the document

for (review, category) in documents:                        # looping through the documents and build the feature set
    # print(review,category)
    feature_set.append((find_features(review),category))    # adding elements to the feature set

# print(feature_set)

training_set = feature_set[:1900]
testing_set = feature_set[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Accuracy : ", nltk.classify.accuracy(classifier,testing_set)*100)
classifier.show_most_informative_features(15)












