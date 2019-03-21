import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
print("HiddenLayerInput ",hidden_layer_input)
hidden_layer_output = sigmoid(hidden_layer_input)
print("HiddenLayerOutput ",hidden_layer_output)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
print("outputLayerInput",output_layer_in)
output = sigmoid(output_layer_in)
print("Output ",output)


## Backwards pass
## TODO: Calculate output error
error = target - output
print("OutputError ",error)

# TODO: Calculate error term for output layer
output_error_term = error * output * (1-output)

# TODO: Calculate error term for hidden layer
hidden_error_term = output_error_term * weights_hidden_output * hidden_layer_output

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = None

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = None

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
