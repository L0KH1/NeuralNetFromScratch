import math
import random
import numpy as np

def setup(inputs, hidden, hiddenSize, actfxn, outputs, costfxn):
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
            "aval": 0.0}])
    
    # now we move to set up our output layer
    schema.append(outputs*[{
        "weights": len(schema[len(schema)-1])*[random.random()],
        "biases": len(schema[len(schema)-1])*[random.random()],
        "aval": 0.0}])
    
    return schema

model = setup((2,2),3,(4,4,4),"ReLU",3,"BCE")

for layer in model:
    print("\n",layer,"\n")