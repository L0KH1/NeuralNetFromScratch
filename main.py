import numpy as np
import pandas as pd

#Input layer

# alright, so, we're going to define our node class

class Node:
    def __init__(self):
        # activation value
        aval=activationValue()
        # nodes which input to it
        inputs =[]
        # nodes to which it outputs
        outputs=[]
        # biases applied to the node
        bias=0
        # node layer - integer
        layer=0
    def activationValue(inputs,squasher):
        #Relu(activations of previous layer x weights of inputs + biases for layer)
        return ReLU(prev*inputs+biases)

# now I'm gonna group together a bunch of the nodes so we can do fun stuff
class Layer:

    # a layer is a list of nodes
    def __init__(self, nodes):
        # put some stuff here
        node=self.node

    
# layer iterable, the index of a layer points to a list of the nodes in that layer

def ReLU(input):
    if input > 0:
        return input
    else:
        return 0