import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x)))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1
rows = inputs.shape[0]
# weightSum = 0
# for i in range(0,rows):
#     weightSum+=inputs[i]*weights[i]
weightSum = np.dot(inputs,weights)
weightSum+=bias

output = sigmoid(weightSum)
print('Output:')
print(output)
