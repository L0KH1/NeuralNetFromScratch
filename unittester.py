import numpy as np
import math
import pandas as pd

def softmax(outputlayer):
    softoutput=[]
    # get the value of our denominator
    denom=0
    for val in outputlayer:
        denom+=math.e**val

    for val in outputlayer:
        softoutput.append(math.e**val/ denom)
    return softoutput

test=[1,2,3,4]

print(softmax(test))