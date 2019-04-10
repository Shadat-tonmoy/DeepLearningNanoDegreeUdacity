import tensorflow as tf
from PreprocessDataset import createFeatureSetAndLabel


train_x, train_y, test_x, test_y = createFeatureSetAndLabel(['dataset/pos.txt','dataset/neg.txt'])


for x in train_x:
    print("OneHotData ",x," Size ",len(x))

num_of_nodes_in_hidden_layer_1 = 500
num_of_nodes_in_hidden_layer_2 = 500
num_of_nodes_in_hidden_layer_3 = 500


num_of_classes = 2                                                      # positive and negative
batch_size = 100                                                        # batch size for training the network

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network_model(data):

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), num_of_nodes_in_hidden_layer_1])),
                      'biases': tf.Variable(tf.random_normal([num_of_nodes_in_hidden_layer_1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hidden_layer_1, num_of_nodes_in_hidden_layer_2])),
                      'biases': tf.Variable(tf.random_normal([num_of_nodes_in_hidden_layer_2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hidden_layer_2, num_of_nodes_in_hidden_layer_3])),
                      'biases': tf.Variable(tf.random_normal([num_of_nodes_in_hidden_layer_3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hidden_layer_3,num_of_classes])),
                    'biases':tf.Variable(tf.random_normal([num_of_classes]))}

    hidden_layer_1_output = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    hidden_layer_1_output = tf.nn.relu(hidden_layer_1_output)

    hidden_layer_2_output = tf.add(tf.matmul(hidden_layer_1_output,hidden_layer_2['weights']),hidden_layer_2['biases'])
    hidden_layer_2_output = tf.nn.relu(hidden_layer_2_output)

    hidden_layer_3_output = tf.add(tf.matmul(hidden_layer_2_output,hidden_layer_3['weights']),hidden_layer_3['biases'])
    hidden_layer_3_output = tf.nn.relu(hidden_layer_3_output)

    output = tf.add(tf.matmul(hidden_layer_3_output,output_layer['weights']),output_layer['biases'])

    return output








