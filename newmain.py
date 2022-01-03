import math
import random
import numpy as np


# define derivatives of transfer functions
def ReLU_Derivative(output):
    if output > 0:
        return 1
    elif output < 0:
        return 0
    else:
        raise ValueError("No derivative when output = 0")

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

def Sigmoid(input):
    return 1/(1+(math.e)**-input)

def Tanh(input):
    return math.e**(2*input)-1/math.e**(2*input)+1

def setupNN(inputs, hidden, hiddenSize, actfxn, outputs, costfxn):
    # the overarching structure
    schema=[]

    # inputs is an nxn-dimensional vector, so we've gotta turn it into an nx1 (or 1xn) vector to input it into our net
    input_length=1
    for element in inputs:
        input_length*=element
    inputs=[0]*input_length

    # the first entry of our schema is a list with length equal to our inputs
    schema.append(inputs)

    # setting up number of hidden layers and number of neurons in each layer
    for i in range(hidden):
        # here we randomly initialize the weights and biases each neuron receives from the previous layer
        schema.append(hiddenSize[i]*[{
            "weights": len(schema[i])*[random.random()], 
            "biases": len(schema[i])*[random.random()], 
            "aval": 0.5}])
    
    # now we move to set up our output layer
    schema.append(len(range(len(outputs)))*[{
        "weights": len(schema[len(schema)-1])*[random.random()],
        "biases": len(schema[len(schema)-1])*[random.random()],
        "aval": 0.5}])
    
    return schema

model = setupNN((2,2),3,(3,5,4),"ReLU",('red','green','blue'),"BCE")

# for layer in model:
#     print("\n",layer,"\n")

# print('\n First hidden layer \n', model[1][0].get('weights'))


def downstream(model):
        # pull up the previous layer and grab the activation values from each neuron, then pull up then multiply each activation value by the weight the neuron in the previous layer mapped to the neuron whose value we're trying to calculate

        # here's a helper function which will make accessing the activation values of the previous layer much easier
        def get_prev_avals(layer):
            avals=[]
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
        for layer in range(1,len(model)):
            # calculate the activation value for each node
            for node in range(len(model[layer])):
                # print('Weights',model[layer][node].get('weights'),'\n')
                # print('Biases',model[layer][node].get('biases'),'\n')
                # print('Prev Act Vals',get_prev_avals(model[layer]),'\n')
                prev_layer=get_prev_avals(model[layer-1])
                # applying softmax to our output layer
                if layer == len(model):
                    model[layer][node]['aval']=softmax(ReLU(np.dot(model[layer][node].get('weights'), prev_layer) + np.dot(model[layer][node].get('biases'), prev_layer)))
                else:
                    model[layer][node]['aval']=ReLU(np.dot(model[layer][node].get('weights'), prev_layer) + np.dot(model[layer][node].get('biases'), prev_layer))
        return model

# after we've run forwards through our netowrk, we need to calculate the cost, and then backpropagate that error

# calculate the cost from a single run through the network of a training sample
# def calculate_cost(model, label, costfxn):
#     cost=0
#     for output in model[len(model)-1]:


# def backpropagate(model):



# testing area

print(downstream(model))

