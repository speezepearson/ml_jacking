import numpy as np
import random

# Code slightly modified from Eric Jorgenson's code

def Experiment1(): 
    heads = tails = 0
    for i in range(0,12):
        res = random.random()
        if res < 0.5:
            tails += 1
        else:
            heads += 1
    return (heads, tails)

def Experiment2(): 
    heads = tails = 0
    while True:
        res = random.random()
        if res < 0.5:
            tails += 1
        else:
            heads += 1
        if heads == 3:
            return (heads, tails)


data1 = []
data2 = []
for i in range(0,100000): 
    data1.append(Experiment1())
    data2.append(Experiment2())

print ("Estimate from experiment(s) 1: ", sum(nh for nh,nt in data1)/sum(nt for nh,nt in data1))
print ("Estimate from experiment(s) 2: ", sum(nh for nh,nt in data2)/sum(nt for nh,nt in data2))
