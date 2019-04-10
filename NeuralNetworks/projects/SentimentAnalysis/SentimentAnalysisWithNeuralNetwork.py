import tensorflow as tf
from PreprocessDataset import createFeatureSetAndLabel


train_x, train_y, test_x, test_y = createFeatureSetAndLabel(['dataset/pos.txt','dataset/neg.txt'])


# for x in train_x:
#     print("OneHotData ",x," Size ",len(x))

num_of_nodes_in_hidden_layer_1 = 800
num_of_nodes_in_hidden_layer_2 = 800
num_of_nodes_in_hidden_layer_3 = 800


num_of_classes = 2                                                      # positive and negative
batch_size = 100                                                        # batch size for training the network

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neuralNetworkModel(data):

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

def trainNeuralNetwork(x):

    prediction = neuralNetworkModel(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_of_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range (num_of_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                epoch_loss += c
                i += batch_size
            print("Epoch ",epoch, " Completed out of ",num_of_epochs, " Training Loss ",epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy : ", accuracy.eval({x:test_x, y:test_y}))


trainNeuralNetwork(x)




















