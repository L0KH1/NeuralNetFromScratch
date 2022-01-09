import math
import random
import numpy as np

# defining our softmax function
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

# definining some transfer/activation functions
def LReLU(input):
    return ReLU(input)-.01*min(0, input)

def ReLU(input):
    if input > 0:
        return input
    else:
        return 0

def sigmoid(input):
    return 1/(1+(math.e)**-input)

def tanh(input):
    return math.e**(2*input)-1/math.e**(2*input)+1

# define derivatives of transfer functions
def ReLU_Derivative(output):
    if output > 0:
        return 1
    elif output < 0:
        return 0
    else:
        raise ValueError("No derivative when output = 0")

def sigmoid_d(output):
    return sigmoid(output)*(1-sigmoid(output))


# one-hot the labels
# will probably be best to make some class or method to handle diverse sets of data
# will need to know all the labels, when given a label, it checks the list of labels, assigns a 1 in the list for the index designating that label (list is initialized to equal all 0s)
def one_hot(labels):
    # so, need to create a vector representation of the labels, where each label has an index, and the input with the label we want
    ohlabels = [0]*len(labels)
    return ohlabels


# initialize our network - for the prototype just using Sigmoid, MSE, and SGD
def setupNN(layers, layersize, activationfxn, costfxn, optimizer):
    # the overarching structure
    schema = []

    # inputs is an nxn-dimensional vector, so we've gotta turn it into a 1 dimensional vector to input it into our net
    input_length = 1 # a temp variable for calculation purposes
    for element in layers:
        input_length *= element
    inputs = [0]*input_length # input is initialized to a 0 vector (change later)

    # the first entry of our schema is a list with length equal to our inputs
    schema.append(inputs) # sets up the input layer

    # setting up number of hidden layers and number of neurons in each layer
    for i in range(hidden):
        # here we randomly initialize the weights and biases each neuron receives from the previous layer
        schema.append(hiddenSize[i]*[{
            "weights": len(schema[i])*[random.random()],
            "biases": len(schema[i])*[random.random()],
            "aval": 0.5,
            "gradients":[]
            }])

    # now we move to set up our output layer
    schema.append(len(range(len(outputs)))*[{
        "weights": len(schema[len(schema)-1])*[random.random()],
        "biases": len(schema[len(schema)-1])*[random.random()],
        "aval": 0.5,
        "gradients":[]}
        ])

    return schema


model = setupNN((2, 2), 3, (3, 5, 4), "ReLU", ('red', 'green', 'blue'), "BCE")

# for layer in model:
#     print("\n",layer,"\n")

# print('\n First hidden layer \n', model[1][0].get('weights'))


def downstream(model):
    # pull up the previous layer and grab the activation values from each neuron, then pull up then multiply each activation value by the weight the neuron in the previous layer mapped to the neuron whose value we're trying to calculate

    # here's a helper function which will make accessing the activation values of the previous layer much easier
    def get_prev_avals(layer):
        avals = []
        for node in layer:
            # used to avoid error while testing without inputs
            if layer is not model[0]:
                # print("Layer: ",layer, "\n\n Node: ",node)
                # print("\nnode.get('aval')",node.get('aval'),'\n')
                avals.append(node.get('aval'))
            else:
                # print('\n\nfirst layer \n\n')
                avals.append(.6)
        return avals

    # for each layer in the model, starting from the first hidden layer
    for layer in range(1, len(model)):
        # calculate the activation value for each node
        for node in range(len(model[layer])):
            # print('Weights',model[layer][node].get('weights'),'\n')
            # print('Biases',model[layer][node].get('biases'),'\n')
            # print('Prev Act Vals',get_prev_avals(model[layer]),'\n')
            prev_layer = get_prev_avals(model[layer-1])
            # applying softmax to our output layer
            if layer == len(model):
                model[layer][node]['aval'] = softmax(sigmoid(np.dot(model[layer][node].get(
                    'weights'), prev_layer) + np.dot(model[layer][node].get('biases'), prev_layer)))
            # hidden layers
            else:
                model[layer][node]['aval'] = sigmoid(np.dot(model[layer][node].get(
                    'weights'), prev_layer) + np.dot(model[layer][node].get('biases'), prev_layer))
    return model

# after we've run forwards through our netowrk, we need to calculate the cost, and then backpropagate that error
# calculate the cost from a single run through the network of a training sample
def calculate_cost(model, label, costfxn):
    cost = 0
    # go through the output layer and add up the costs
    for i in range(len(label)):
        cost += costfxn(output[len(model-1)][i].get('aval'), label[i])
    # get the average cost
    cost /= len(label)
    return cost

# now time for back-propagation! - an automatic differentiation method which allows us to calculate the error correction for each weight and bias!
def backpropagate(model):
    # moving from the last layer to the first layer
    for layer in reversed(range(1, len(model))):
        # moving through the nodes
        for node in range(len(model[layer])):
            # want to calculate change relative to weights and biases
            # new weight = old weight - learning rate * (input neuron * (prediction-actual)*output weight)
            
    return model

def stochasticgradientdescent(model,lr):
    # moving from the last layer to the first layer
    for layer in reversed(range(1, len(model))):
        # moving through the nodes
        for node in range(len(model[layer])):
            # want to calculate change relative to weights and biases
            # new weight = old weight - learning rate * (input neuron * (prediction-actual)*output weight)
            model[layer][node]['aval']=model[layer][node].get('aval')-lr*gradient

# testing area
print(downstream(model))
