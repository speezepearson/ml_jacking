import numpy as np
import random

# Code slightly modified from Eric Jorgenson's code

def Experiment1(): 
    tails = 0
    for i in range(0,12):
        res = random.random()
        if res < 0.5:
            tails += 1
    return tails / 12.0

def Experiment2(): 
    tails = 0
    length = 0
    while True:
        res = random.random()
        length += 1
        if res < 0.5:
            tails += 1
            if tails >= 3:
                return 3.0 / length    

    return -1000000000000


data1 = []
data2 = []
for i in range(0,100000): 
    data1.append(Experiment1())
    data2.append(Experiment2())

print ("Estimate from experiment(s) 1: ", np.mean(data1))
print ("Estimate from experiment(s) 2: ", np.mean(data2))
