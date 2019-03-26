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
        'biases': tf.Variable(tf.random_normal([num_of_nodes_in_hl1]))
    }

    hidden_layer_2 = {
        'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hl1, num_of_nodes_in_hl2])),
        'biases': tf.Variable(tf.random_normal([num_of_nodes_in_hl2]))
    }

    hidden_layer_3 = {
        'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hl2, num_of_nodes_in_hl3])),
        'biases': tf.Variable(tf.random_normal([num_of_nodes_in_hl3]))
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([num_of_nodes_in_hl3, num_of_classes])),
        'biases': tf.Variable(tf.random_normal([num_of_classes]))
    }

    # computation performed in each layer
    # (input_data * weights) + biases

    layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.add(tf.matmul(layer_3,output_layer['weights']),output_layer['biases'])

    return output

def train_neural_network(x):

    prediction = neural_network_model(x) # getting the predicted value for the input data  x

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction)) # cross entropy with logits is cost function to
    # calculate the difference between the prediction and  the known label in the dataset. Boths are in one hot format

    optimizer = tf.train.AdamOptimizer().minimize(cost) # using adam optimizer minimize the cost

    num_of_epochs = 10      # number of epoch

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables()) #start a tf session to perform computatitonal operation

        for epoch in range(num_of_epochs): #looping through each epoch to train the network
            epoch_loss = 0

            # looping through the number of data to train in a single epoch. Here _ indicates a variable that we don't care about its name. Dividing total number of example by batch size we get the number of training example in a single epoch.
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # getting the splitted training example with label. These preprocessing is already done by mnist. Real world data need to be preprocessed.
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                c,_ = sess.run([cost,optimizer],feed_dict={x:epoch_x,y:epoch_y})

                #adding the cost denoted by c to epoch loss after each epoch
                epoch_loss += c
            print("Epoch ",epoch,' Completed out of ',num_of_epochs,' Epoch Loss ',epoch_loss)

        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print("Accuracy ",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)




















