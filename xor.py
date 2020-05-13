"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh, RELU


inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

#  np.random.seed(664299)

# NOTE: RELU works or does not work based on the PRNG random state which
# affects the initial condition of the weights in the neural network.
# Sometimes it is as bad as linear layers when the number of neurons are the
# same as the number of inputs / features
hidden_layer_neurons = 8

net = NeuralNet([
    Linear(input_size=2, output_size=hidden_layer_neurons),
    #  Tanh(),
    RELU(),
    Linear(input_size=hidden_layer_neurons, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
