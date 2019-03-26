import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # self.activation_function = lambda x : 1/(1+np.exp(x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
           return 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        # print("Delta Input to hidden weight ",delta_weights_i_h.shape)
        # print("Delta Hidden to output weight ",delta_weights_h_o.shape)
        # print("Feature Shape",features," Shape ",features.shape)
        # print("Target Shape",targets," Shape ",targets.shape)
        for X, y in zip(features, targets):
            X = np.array(X, ndmin=2)
            y = np.array(y, ndmin=2)
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            # print("After Forward Pass Final Layer Output ",final_outputs," Shape ",final_outputs.shape," Hidden Layer Output",hidden_outputs," Shape ",hidden_outputs.shape)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
        # hidden_inputs = hidden_inputs.reshape(1,2)
        # print("In forward pass input  ",X," Shape ",X.shape," Weight Input to Hidden ",self.weights_input_to_hidden," Shape ",self.weights_input_to_hidden.shape)
        # print("In forward pass hidden_inputs ",hidden_inputs," Shape ",hidden_inputs.shape)
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        # print("Forward pass hidden unit output ",hidden_outputs," Shape ",hidden_outputs.shape)

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        # print("Forward Pass Final Layer Input ",final_inputs," Shape ",final_inputs.shape)
        final_outputs = final_inputs # signals from final output layer
        # print("Forward Pass Final Layer Output ",final_outputs," Shape ",final_outputs.shape)

        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement 03.Backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        # print("In Back propagation final layer output ",final_outputs," Shape ",final_outputs.shape,"\nHidden Output ",hidden_outputs," Shape ",hidden_outputs.shape,"\nFeature Input ",X," Shape ",X.shape," Change in i_h ",delta_weights_i_h," Shape ",delta_weights_i_h.shape,"\nChange is h_o ",delta_weights_h_o," Shape ",delta_weights_h_o.shape)

        # TODO: Output error - Replace this value with your calculations.
        error = y-final_outputs # Output layer error is the difference between desired target and actual output.
        # print("In Backprop final layer error ",error," Shape ",error.shape)

        f_prime_output = final_outputs*(1-final_outputs)


        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error * 1
        # print("Output layer error term ",output_error_term," Shape ",output_error_term.shape)

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term,self.weights_hidden_to_output.T)
        # print("Weight hidden to output ",self.weights_hidden_to_output.T," Shape ",(self.weights_hidden_to_output.T).shape)
        # print("Output Error Term ",output_error_term," Shape ",output_error_term.shape)
        # print("Hidden error",hidden_error)
        # hidden_error = hidden_error.reshape(2,1)
        # print("Hiddent layer error ",hidden_error," Shape ",hidden_error.shape)

        f_prime_hidden_output = hidden_outputs*(1-hidden_outputs)
        # f_prime_hidden_output = f_prime_hidden_output.reshape(1,2)
        # print("F prime hidden output ",f_prime_hidden_output," Shape ",f_prime_hidden_output.shape)


        hidden_error_term = hidden_error*f_prime_hidden_output
        # print("Hidden Layer Error Term ",hidden_error_term," Shape ",hidden_error_term.shape)
        # print("Input Layer ",X," Shape ",X.shape)
        # print("Input Layer X[:None] ",X[:,None]," Shape ",X[:,None].shape)

        # print("Change in i_h weight ",(hidden_error_term*X[:,None])," Shape ",(hidden_error_term*X[:,None]).shape)
        # print("Delta Weight in i_h ",delta_weights_i_h," Shape ",delta_weights_i_h.shape)

        # print("Output Error Term ",output_error_term," Shape ",output_error_term.shape,"\nHidden Layer Output ",hidden_outputs," Shape ",hidden_outputs.shape," Delta Weight H2O ",delta_weights_h_o," Shape ",delta_weights_h_o.shape)
        # print("Change in output layer weight ",output_error_term*hidden_outputs.T," Shape ",(output_error_term*hidden_outputs.T).shape)
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term*X.T
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term*hidden_outputs.T
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output = self.weights_hidden_to_output + (self.lr*delta_weights_h_o / n_records) # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden = self.weights_input_to_hidden + (self.lr * delta_weights_i_h / n_records) # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features,self.weights_input_to_hidden)# signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# iterations = 100
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1

iterations = 5000
learning_rate = 0.3
hidden_nodes = 40
output_nodes = 1