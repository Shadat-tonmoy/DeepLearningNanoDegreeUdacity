import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import pickle
from collections import Counter

lemmitizer = WordNetLemmatizer()

num_of_lines = 10000000

def createLexicon(data_set):
    lexicon = []
    final_lexicon = []
    for file in data_set:
        with open(file, 'r') as f:
            contents = f.readlines()
            for line in contents[:num_of_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
    lexicon = [lemmitizer.lemmatize(i) for i in lexicon]

    word_counts = Counter(lexicon)

    for word in word_counts:
        if 1000 > word_counts[word] > 50:
            final_lexicon.append(word)



def handleSampleData(sampleData, lexicon, classification):

    '''
    featureSet =
    [
        [[0,0,0,1,1,0,1,1,1,0,0,0],[1,0]].
        [[0,1,0,0,0,1,1,1,1,0,0,0],[0,1]],
        [[features],[classification]]
    ]
    :param sampleData:
    :param lexicon:
    :param classification:
    :return:
    '''

    featureSet = []

    with open(sampleData,'r') as f:
        contents = f.readlines()

        for line in contents[:num_of_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmitizer.lemmatize(i) for i in current_words]
            features = np.zeros()

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1

            features = list(features)
            featureSet.append([features,classification])

    return featureSet


def createFeatureSetAndLabel(dataset,test_size = 0.1):
    lexicon = createLexicon(dataset)

    features = []

    features+=handleSampleData('dataset/pos.txt', lexicon, [1, 0])
    features+=handleSampleData('dataset/neg.txt', lexicon, [0, 1])

    random.shuffle(features)

    testing_size = int(test_size*len(features))

    features = np.array(features)

    train_x = list(features[:, 0][:-testing_size])

    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])

    test_y = list(features[:, 1][-testing_size:])

    return train_x,train_y,test_x,test_y

















