import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
steps 

activation function = (relu/sigmoid)

Feed Forward Network : 
input -> weight -> hidden layer 1 -> activation function  -> hidden layer 2 -> activation function -> hidden layer 3 -> 
activation function -> weight -> output layer  

Error Calculation :
compare predicted output  with actual output using cost or loss function

Error Minimization : 
minimize the cost using optimization function like adam optimizer or gradient decent optimizer

Back Propagation : 
each optimization function use back-propagation in the underneath to update the weights

Feed Forward + Back Propagation = Epoch
'''

mnist = input_data.read_data_sets("tmp/data/",one_hot=True) #reading data from mnist dataset

num_of_nodes_in_hl1 = 500   #number of nodes in hidden layer 1
num_of_nodes_in_hl2 = 500   #number of nodes in hidden layer 2
num_of_nodes_in_hl3 = 500   #number of nodes in hidden layer 3

num_of_classes = 10         #number of nodes in output layer or number of predicted classes
batch_size = 100            #size of each batch to train the network

#matrix shape is defined as height x width or row x column
x = tf.placeholder('float',[None,784])  #x is the input value defined as a matrix. Here number of image as input is not defined. That is indicaded by 'None' but each input has 784 (28x28) pixels value as feature

y = tf.placeholder('float') #y is the targetted value

def neural_network_model(data):

    hidden_layer_1 = {
        'weights': tf.Variable(tf.random_normal([784,num_of_nodes_in_hl1])),
        'biases': tf.Variable(tf.random_normal(num_of_nodes_in_hl1))
    }

    hidden_layer_2 = {
        'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hl1, num_of_nodes_in_hl2])),
        'biases': tf.Variable(tf.random_normal(num_of_nodes_in_hl2))
    }

    hidden_layer_3 = {
        'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hl2, num_of_nodes_in_hl3])),
        'biases': tf.Variable(tf.random_normal(num_of_nodes_in_hl3))
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hl3, num_of_classes])),
        'biases': tf.Variable(tf.random_normal(num_of_classes))
    }

    # computation performed in each layer
    # (input_data * weights) + biases

    layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output= tf.add(tf.matmul(layer_3,output_layer['weights']),output_layer['biases'])

    return output



