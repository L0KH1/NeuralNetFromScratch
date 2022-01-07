import random as random
import math

class NN:
    def __init__(self, layers, layersize, activationfxn, costfxn, optimizer):
        self.layers=layers # integer
        self.layersize=layersize # tuple
        # activationfxn is a tuple of length 1 or length=layers, to allow for different activation function at each layer
        if len(activationfxn) < len(layers):
            self.activationfxn=activationfxn[0] # if not enough layers are entered, just use the first activation fxn entered
        else:
            self.activationfxn=activationfxn 
        self.costfxn=costfxn # MSE for now
        self.optimizer=optimizer # basic SGD for now

class Node:
    def __init__(self, )