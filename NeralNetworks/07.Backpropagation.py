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
hidden_layer_output = sigmoid(hidden_layer_input)

print ("Hidden Layer input")
print (hidden_layer_input)
print ("Hidden Layer Output")
print (hidden_layer_output)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)
print ("Output Layer Input")
print (output_layer_in)
print ("Output Layer Output")
print (output)
## Backwards pass
## TODO: Calculate output error
error = target - output

sigmoid_prime_output_layer = output * (1 - output)
sigmoid_prime_hidden_layer = hidden_layer_output * (1 - hidden_layer_output)
# TODO: Calculate error term for output layer
output_layer_error_term = error * sigmoid_prime_output_layer
print ("Output layer Error Term ",output_layer_error_term)

# TODO: Calculate error term for hidden layer
hidden_layer_error_term = [output_layer_error_term * weights_hidden_output[0] * sigmoid_prime_hidden_layer[0],
                           output_layer_error_term * weights_hidden_output[1] * sigmoid_prime_hidden_layer[1]]
print (hidden_layer_error_term)
# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_layer_error_term * hidden_layer_output
print ("Delta of output to hidden layer")
print (delta_w_h_o)
print ("Hidden Layer Error Term ")
print (hidden_layer_error_term)
# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = list()

for xTerm in x:
    unitError = list()
    for errorTerm in hidden_layer_error_term:
        err = learnrate * xTerm * errorTerm
        unitError.append(err)
    delta_w_i_h.append(unitError)

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
