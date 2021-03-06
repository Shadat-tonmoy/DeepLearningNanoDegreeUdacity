import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import pickle
from collections import Counter

lemmitizer = WordNetLemmatizer()                                # nltk library function to lemmatize or step towards the root word

num_of_lines = 10000000


def createLexicon(data_set):                                    # function to create lexicon that is the most common words available in the dataset
    '''

    :param data_set:
    :return final_lexicon:
    '''
    lexicon = []                                                # empty list of lexicon
    final_lexicon = []                                          # list of final most common words
    for file in data_set:                                       # looping through all the files in the dataset
        with open(file, 'r') as f:                              # open a specific file like  Pos.txt or Neg.txt to read
            contents = f.readlines()                            # read the contents of a file line by line and store all the lines into an array
            for line in contents[:num_of_lines]:                # loop through the top/first num_of_lines lines from all the lines
                all_words = word_tokenize(line.lower())         # tokenize the words from each line using nltk word tokenizer
                lexicon += list(all_words)                      # add the list of tokenized word from a single line into the list of lexicon
    lexicon = [lemmitizer.lemmatize(i) for i in lexicon]        # lematizing all the words added into the list of lexicon

    word_counts = Counter(lexicon)                              # frequency counter for each of the words in the lexicon set like {'the':1234,'and':345}

    for word in word_counts:                                    # looping through all the words in the word counter
        if 1000 > word_counts[word] > 50:                       # take only the words with frequency more then 50 and less then 1000
            final_lexicon.append(word)                          # adding the matched word into the final lexicon list

    print("Size of lexicon ",len(final_lexicon))

    return final_lexicon


def handleSampleData(sampleData, lexicon, classification):      # function to preprocess the sample data based on the lexicon available

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
    :return featureSet:
    '''

    featureSet = []                                                             # empty list of feature for this current sample data

    with open(sampleData,'r') as f:                                             # opening the sample data to read
        contents = f.readlines()                                                # read the contents of the  file and store the lines into an array

        for line in contents[:num_of_lines]:                                    # read the top most number_of_lines line from the list of all lines
            current_words = word_tokenize(line.lower())                         # tokenize all the word from each line using word tokenizer
            current_words = [lemmitizer.lemmatize(i) for i in current_words]    # lemmatizing all the word from each line using wordnet lemmatizer
            features = np.zeros(len(lexicon))                                   # setting default zero against all the value available in the lexicon. This indicates are they
                                                                                # present or absent in the sample data

            for word in current_words:                                          # looping through all the words for a specific line of a sample data to check whether it is
                if word.lower() in lexicon:                                     # present or not in the lexicon
                    index_value = lexicon.index(word)                           # getting the index of available word from the lexicon list
                    features[index_value] += 1                                  # updating the default zero value to 1 or more based on the occurrence of that word. The mapping
                                                                                # key here is the index value. That is if the 7th word in the lexicon list is present then the
                                                                                # 7th index in the features map is updated. The length of feature map and lexicon map is similar

            features = list(features)                                           # making list of all the features
            featureSet.append([features,classification])                        # appending list of features and their label as Pos [1,0] or Neg [0,1] into the feature set

    return featureSet                                                           # returning the feature set

def createFeatureSetAndLabel(dataset,test_size = 0.1):                          # function to merge all the function defined previously to built feature and label from dataset

    lexicon = createLexicon(dataset)                                            # create list of lexicon from the given dataset

    features = []                                                               # empty list of features and label from all the dataset

    features+=handleSampleData('dataset/pos.txt', lexicon, [1, 0])              # adding features and labels from positive sentiment dataset into features list
    features+=handleSampleData('dataset/neg.txt', lexicon, [0, 1])              # adding features and labels from negative sentiment dataset into features list

    random.shuffle(features)                                                    # shuffle the features to train the model with more random data

    features = np.array(features)                                               # converting the features list into numpy array

    testing_size = int(test_size*len(features))                                 # getting the actual size of test data from the features list size

    train_x = list(features[:, 0][:-testing_size])                              # extracting the training data samples from the features set. [:,0] means getting the 0th element
                                                                                # from all the elements of the list. 0th element is the list of features for all the data. Here -
                                                                                # indicates the indexes of last testing sizes
    train_y = list(features[:, 1][:-testing_size])                              # extracting the training label samples from the features set. [:,1] means getting the 1th element
                                                                                # from all the elements of the list. 1th element is the list of labels for all the data.
    test_x = list(features[:, 0][-testing_size:])                               # extracting the test data

    test_y = list(features[:, 1][-testing_size:])                               # extracting the test labels

    return train_x,train_y,test_x,test_y


if __name__ == 'main':

    train_x,train_y,test_x,test_y = createFeatureSetAndLabel(['dataset/pos.txt','dataset/neg.txt'])
    with open("output/sentiment_analysis.pickle","wb") as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
















