import math
import random

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

# dot product helper function
def dot(v1,v2):
    dotted=[]
    for val in range(len(v1)):
        dotted.append(v1[val]*v2[val])
    return dotted

# initialize our network - for the prototype just using Sigmoid, MSE, and SGD
def setupNN(input, layers, afxn="sigmoid", cfxn="mse", ofxn="sgd"):
    # the overarching structure
    schema = []

    # setting up number of hidden layers and number of neurons in each layer
    for i in range(len(layers)): # looping over each layer
        if i == 0: # when setting up the input layer
            schema.append( # add layer to schema, contents described below
                [ # store it in a list so we can iterate over it
                { # an empty dictionary, later on need to fill this up with data
                "aval": 1,
                }
                for i in range(layers[i])
                ]
            )
        else: # when setting up the rest of the layers - differs only by storing gradients for weights
            schema.append( # add layer to schema, contents described below
                [{ # the number of nodes in the layer * a dictionary containing all of the attributes of a neuron
                "weights": len(schema[i-1])*[random.random()], # weights=#neurons in previous layer, initialized to a random value. Note, the weights stored by each node in each layer are those that are coming into it
                "bias": 1, # each neuron has a bias
                "aval": 0.5, # activation value, populated later by forwardprop
                "wgradients": len(schema[i-1])*[0], # gradients for weights, populated later by backprop
                "bgradient": 0# gradient for bias, populated later by backprop
                }
                for i in range(layers[i])
                ]
            )
    return schema

network = setupNN(0,(3,2,2)) #initialize our network structure

# this provides a nice visualization of what the network looks like
for layer in range(len(network)):
    print('Layer ',layer,'---------------\n')
    for node in range(len(network[layer])):
        print('    Node ',node,':',network[layer][node],'\n')


# calculate the activation values for the model
def forwardprop(model):
    
    # testing area - hardcoding in known values to see if data flowing properly

    # activation values of input neurons
    model[0][0]['aval']=1
    model[0][1]['aval']=4
    model[0][2]['aval']=4

    # first layer of weights
    model[1][0]['weights']=[.1,.3,.5]
    model[1][1]['weights']=[.2,.4,.6]
    
    # second layer of weights
    model[2][0]['weights']=[.7,.9]
    model[2][1]['weights']=[.8,.1]

    # a little helper function for grabbing the activations and biases for the previous layer
    def plab(layer):
        activations=[]
        biases=[]
        ab=[]
        for index in layer:
            activations.append(layer[index].get('aval'))
            biases.append(layer[index].get('bias'))
        ab.append(activations)
        ab.append(biases)
        return ab

    for layer in range(len(model)): # loop over each layer in the model
        if layer == 0: #we don't change the value of our input layer's nodes
            break
        else: # all layers that aren't the input layer
            for node in range(len(model[layer])): # loop over each node in the layer
                model[layer][node]["aval"]= sigmoid(dot(model[layer][node].get("weights"),plab(model[layer-1][node])[0]) + plab(model[layer-1][node])[1]) # the activation function
    return model

network = forwardprop(network) # running a first forward propagation to see if we get some good values
print('\n\n **************** After Forward Prop ****************\n\n')
for layer in range(len(network)):
    print('Layer ',layer,'---------------\n')
    for node in range(len(network[layer])):
        print('    Node ',node,':',network[layer][node],'\n')
print(network[2][0].get('weights'),'\n',network[2][1].get('weights'))