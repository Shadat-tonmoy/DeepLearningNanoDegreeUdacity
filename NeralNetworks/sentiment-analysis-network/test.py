import time
import sys
import numpy as np
from collections import Counter

g = open('reviews.txt', 'r')  # What we know!
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt', 'r')  # What we WANT to know!
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()


class TestNetword:
    def __init__(self):
        self.readData()
        self.countWordFreq()
        self.calculateWordRatio()
        self.calculateLogOfRations()
        self.buildVocab()
        self.layer_0 = np.zeros((1, self.vocabSize))
        self.buildWordIndex()
        self.printStatus()

    def readData(self):
        g = open('reviews.txt', 'r')  # What we know!
        self.reviews = list(map(lambda x: x[:-1], g.readlines()))
        g.close()

        g = open('labels.txt', 'r')  # What we WANT to know!
        self.labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
        g.close()

    def countWordFreq(self):
        self.positive_counts = Counter()
        self.negative_counts = Counter()
        self.total_counts = Counter()
        for i in range(len(self.reviews)):
            self.words = self.reviews[i].split(" ")
            if self.labels[i] == "POSITIVE":
                for word in self.reviews[i].split(" "):
                    self.positive_counts[word] += 1
                    self.total_counts[word] += 1
            else:
                for word in self.reviews[i].split(" "):
                    self.negative_counts[word] += 1
                    self.total_counts[word] += 1

    def calculateWordRatio(self):
        self.pos_neg_ratios = Counter()
        for word, count in list(self.total_counts.most_common()):
            if count >= 100:
                self.pos_neg_ratios[word] = self.positive_counts[word] / float(self.negative_counts[word] + 1)

    def calculateLogOfRations(self):
        for word, ratio in self.pos_neg_ratios.most_common():
            if ratio > 1:
                self.pos_neg_ratios[word] = np.log(ratio)
            else:
                self.pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

    def buildVocab(self):
        self.vocab = set()
        for review in self.reviews:
            for word in review.split(" "):
                self.vocab.add(word)
        self.vocabSize = len(self.vocab)

    def buildWordIndex(self):
        self.word2index = {}
        for i, word in enumerate(self.vocab):
            self.word2index[word] = i

    def printStatus(self):
        print("Total Reviews ", len(self.reviews))
        print("Input Layer Shape ", self.layer_0.shape)

    def updateInputLayer(self, review):
        # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
        word_freq_counter = Counter()

        for word in review.split(" "):
            word_freq_counter[word] += 1
            self.layer_0[0][self.word2index[word]] = word_freq_counter[word]

    def getTargetForLabel(self, label):
        if label == "POSITIVE":
            return 1
        else:
            return 0


# testNet = TestNetword()


# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        # print(self.review_vocab)
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

        self.train(reviews,labels)

    def pre_process_data(self, reviews, labels):

        review_vocab = set()
        # populate review_vocab with all of the words in the given reviews
        for review in reviews:
            for word in review.split(' '):
                review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        # There is no need to split the labels because each one is a single word.
        for label in labels:
            label_vocab.add(label)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # print(reviews)
        # print(labels)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # print(self.word2index)

            # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab,
        #       but for self.label2index and self.label_vocab instead
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

        # print(self.label2index)

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((input_nodes, hidden_nodes))

        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.zeros((hidden_nodes, output_nodes))

        # TODO: Create the input layer, a two-dimensional matrix with shape
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1, input_nodes))

        # print(self.weights_0_1)
        # print(self.weights_1_2)

    def update_input_layer(self, review):
        # TODO: You can copy most of the code you wrote for update_input_layer
        #       earlier in this notebook.
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"

        # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
        word_freq_counter = Counter()

        for word in review.split(" "):
            word_freq_counter[word] += 1
            self.layer_0[0][self.word2index[word]] = word_freq_counter[word]
        # print(self.word2index)
        # print("TotalWordInReview ",len(review.split(" ")))
        print(self.layer_0[0])

    def get_target_for_label(self, label):
        # TODO: Copy the code you wrote for get_target_for_label
        #       earlier in this notebook.
        if label == "POSITIVE":
            return 1
        else:
            return 0

    def sigmoid(self, x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        # TODO: Return the derivative of the sigmoid activation function,
        #       where "output" is the original output from the sigmoid fucntion
        return output * (1 - output)

    def train(self, training_reviews, training_labels):

        # make sure out we have a matching number of reviews and labels
        assert (len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # TODO: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]

            print(review)
            print(label)
            self.update_input_layer(review)

            # TODO: Implement the forward pass through the network.
            #       That means use the given review to update the input layer,
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            #
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.

            # update_input_layer(review)
            # output = get_target_for_label(label)

            # TODO: Implement the back propagation pass here.
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you
            #       learned in class.

            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction
        #             might come from anywhere, so you should convert it
        #             to lower case prior to using it.

        # TODO: The output layer should now contain a prediction.
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`,
        #       and `NEGATIVE` otherwise.
        pass

mlp = SentimentNetwork(reviews[0:1],labels[0:1])