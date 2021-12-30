import numpy as np
import pandas as pd
import math

# Input layer

# alright, so, we're going to define our node class


class Node:
    def __init__(self):
        node = {
            # activation value
            aval: 0.0,
            # weights
            weights: []
            # biases applied to the node
            bias = 0
        }

    def activationValue(inputs, squasher):
        # Relu(activations of previous layer x weights of inputs + biases for layer)
        return ReLU(prev*inputs+biases)

# now I'm gonna group together a bunch of the nodes so we can do fun stuff


class NeuralNet:

    # a layer is a list of nodes
    def __init__(self, inputs, hidden, hiddenSize, actfxn, outputs, labels, costfxn):
        # this is our big papa structure
        schema = []
        # inputs is an n-dimensional vector, so we've gotta flatten it before it can be used in our net
        inputs = inputs.flatten()
        # the first entry of our schema is a list of length equal to our inputs
        schema.append(inputs)
        # now I'm inserting the amount of hidden layers we have and their respective lengths into the schema
        for layer, size in range(hidden), hiddenSize:
            schema.append([0]*size)
        # we setup the output layer
        schema.append(outputs)

    # this function runs our calculations forward through the network

    def right(self):
        # we go layer by layer, and neuron by neuron
        for layer in len(layers):
            for node in len(layer):
                node = node.getActivationValue()

    # this function runs our backpropagation
    def left(self):
        # look at each neuron in the layer, take partial derivative of each incoming weight to see how that affects cost

        # layer iterable, the index of a layer points to a list of the nodes in that layer


def MSE(outputlayer, label):
    cost = 0
    for i in range(len(outputlayer)):
        cost += (outputlayer[i]-label[i])**2
    avgCost = 1.0/len(label)*cost


def BCEL(outputlayer, label):
    cost = 0
    for i in range(len(outputlayer)):
        cost += outputlayer[i]*log(math.e-15+label[i])
    avgCost = 1.0/len(label)*cost
    return -avgCost

# define the softmax function


def softmax(outputlayer):
    softoutput = []
    # get the value of our denominator
    denom = 0
    for val in outputlayer:
        denom += math.e**val

    # get our softmaxed values
    for val in outputlayer:
        softoutput.append(math.e**val / denom)
    return softoutput


# definining some activation functions
def LReLU(input):
    return ReLU(input)-.01*min(0, input)


def ReLU(input):
    if input > 0:
        return input
    else:
        return 0


def Sigmoid(input):
    return 1/(1+(math.e)**-input)


def Tanh(input):
    return math.e**(2*input)-1/math.e**(2*input)+1
