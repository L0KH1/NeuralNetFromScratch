import numpy as np
import random
import math

# Input layer

# alright, so, we're going to define our node class


class Node:
    def __init__(self, layer):
        self.aval = 0.0
        weights = [random.random() for x in range(len(previous_layer))]
        bias = 0.0  # is there one bias per layer?

    def getWeights(self):
        # when we use SGD with backprop we'll adjust the weights according to this optimization algo

        # this is applied starting at the first input layer
    def activationValue(self, inputs, actxn):
        # Relu(activations of previous layer x weights of inputs + biases for layer)
        # pull up the previous layer and grab the activation values from each neuron, then pull up then multiply each activation value by the weight the neuron in the previous layer mapped to the neuron whose value we're trying to calculate
        for weight in weights:
            self.aval += weight*prevlayerneuronaval+bias
        return actfxn(self.aval)

# now I'm gonna group together a bunch of the nodes so we can do fun stuff


class NeuralNet:

    # a layer is a list of nodes
    def __init__(self, inputs, hidden, hiddenSize, actfxn, outputs, labels, costfxn):
        # this is our big papa structure
        self.schema = []
        # inputs is an n-dimensional vector, so we've gotta flatten it before it can be used in our net
        inputs = inputs.flatten()
        # the first entry of our schema is a list of length equal to our inputs
        self.schema.append(inputs)
        # now I'm inserting the amount of hidden layers we have and their respective lengths into the schema
        for layer, size in range(hidden), hiddenSize:
            self.schema.append([0]*size)
        # we setup the output layer
        self.schema.append(outputs)

    # this function runs our calculations forward through the network

    def right(self):
        # we go layer by layer, and neuron by neuron
        for l in range(1, len(self.schema)):
            for n in len(self.schema[i]):
                schema[l][n] = getActivationValue(schema[l][n])

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
