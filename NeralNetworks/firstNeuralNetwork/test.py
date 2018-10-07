import numpy as np
from firstNeuralNetwork.my_answers import NeuralNetwork
from firstNeuralNetwork.actual_answer import NeuralNetworkAnswer

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

network = NeuralNetwork(3, 2, 1, 0.5)
network.weights_input_to_hidden = test_w_i_h.copy()
network.weights_hidden_to_output = test_w_h_o.copy()
# print(network.weights_input_to_hidden)  #(3,2)
# print(network.weights_hidden_to_output)  #(2,1)
# print(inputs.shape)
network.train(inputs,targets)
