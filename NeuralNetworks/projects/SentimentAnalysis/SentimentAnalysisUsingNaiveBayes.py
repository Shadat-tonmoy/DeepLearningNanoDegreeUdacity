import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,NuSVC,LinearSVC
import pickle
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

# classifier = nltk.NaiveBayesClassifier.train(training_set)

saved_classifier = open("naivebayes.pickle","rb")           # opening the saved classifier in read bytes mood
classifier = pickle.load(saved_classifier)                  # load the classifier from saved classifier
saved_classifier.close()                                    # close the saved  classifier file



print("NLTK Naive Bayes Alogrithm Accuracy : ", nltk.classify.accuracy(classifier,testing_set)*100)
classifier.show_most_informative_features(15)

# save_classifier = open("naivebayes.pickle","wb")    # open a pickle file named naivebayes in write bytes mode to save classifier
# pickle.dump(classifier,save_classifier)             # dump the classifier module to pickle file
# save_classifier.close()                             # close pickle file after saving

MNB_Classifier = SklearnClassifier(MultinomialNB())
MNB_Classifier.train(training_set)
print("Multinomial Naive Bayes Algorithm Accuracy :",nltk.classify.accuracy(MNB_Classifier,testing_set)*100)

# GaussianNB_Classifier = SklearnClassifier(GaussianNB())
# GaussianNB_Classifier.train(training_set)
# print("Gaussian Naive Bayes Algorithm Accuracy : ",nltk.classify.accuracy(GaussianNB_Classifier,testing_set)*100)

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print("Bernoullin Naive Bayes Algorithm Accuracy : ",nltk.classify.accuracy(BernoulliNB_Classifier,testing_set)*100)


# LogisticRegression
# SGDClassifier
# SVC
# NuSVC
# LinearSVC


LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("Logistic Regression Classifier Algorithm Accuracy : ",nltk.classify.accuracy(LogisticRegression_Classifier,testing_set)*100)

SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGC Classifier Algorithm Accuracy : ",nltk.classify.accuracy(SGD_Classifier,testing_set)*100)

SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC Classifier Algorithm Accuracy : ",nltk.classify.accuracy(SVC_Classifier,testing_set)*100)


NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC Classifier Algorithm Accuracy : ",nltk.classify.accuracy(NuSVC_Classifier,testing_set)*100)

LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC Classifier Algorithm Accuracy : ",nltk.classify.accuracy(LinearSVC_Classifier,testing_set)*100)

















