#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:00:58 2018

@author: hielke
"""

import numpy as np 
from hopfield import Hopfield
import random 
from pprint import PrettyPrinter as pp
pprint = pp().pprint
from custom_data_structs import Dataset
from plotters import digitFrame
from os import path 

rate = .001
err = .01


matrix = np.array([[1, 1, -1, 1, -1, 1, -1, -1],
                       [1, 1, -1, 1, -1, -1, -1, 1],
                       [1, 1, 1, 1, -1, -1, -1, -1],
                       [-1, 1, -1, 1, -1, 1, -1, 1]])

matrix = np.array([[1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, -1]])

ds = Dataset(path.join('data', 'train_digits.txt'))
matrix = np.array(ds)
A = matrix < 50
matrix = np.ones(np.shape(matrix), dtype=int)
matrix[A] = -1

matrix = np.array([[1, 1, -1, 1, -1, 1, -1, -1],
                       [1, 1, -1, 1, -1, -1, -1, 1],
                       [1, 1, 1, 1, -1, -1, -1, -1],
                       [-1, 1, -1, 1, -1, 1, -1, 1]])

#matrix = np.array([[1, 1, -1, 1, -1, 1, -1, -1],
#                       [1, 1, -1, 1, -1, -1, -1, 1],
#                       [1, 1, 1, 1, -1, -1, -1, -1],
#                       [-1, 1, -1, 1, -1, 1, -1, 1]])

#matrix = np.array([[-1, 1, -1, -1, -1]])

def transformation(vector):
    size = len(vector)
    tres = np.ones((1,size))
    z = np.zeros((size, size + size*(size-1)//2))
    for i in range(size):
        ran = (size-1)*size//2 - (size-1-i)*(size-i)//2
        z[i,ran:ran+size-i-1] = vector[i+1:]
        z[i,-(size-i)] = tres[0,i]
        for j in range(i):
            z[i, ]
        j = i
        while j > 0:
            ran = (size-1)*size//2 -(size-j)*(size-j+1)//2
            z[i, i-1+ran] = vector[j-1]
            j -= 1
    return z

def hopfield_perceptron(matrix):
    
    vector = matrix[0, :]
    z = transformation(vector)
    sol = np.array([random.choice([-.01, .01]) for _ in range(len(z[0,:]))])
    
    amm_vectors = len(matrix[:, 0])
    
    for i in range(1000):
        print("Iteration : %d" % i)
        still_wrong = False
        for j in range(amm_vectors):
            vector = matrix[j, :]
            z = transformation(vector)
            for i in range(len(vector)):
                if not sol.dot(z[i, :]) * vector[i] > err: # prediction incorrect
                    sol += rate * z[i, :] * vector[i]
                    still_wrong = True
        if not still_wrong:
            print("We are good")
            break
        
    triu = zip(*np.triu_indices(len(vector), 1))
    weights = np.matlib.zeros((len(vector), len(vector)))
    for ind, iu in enumerate(triu):
        weights[iu] = sol[ind]
    
    weights += weights.T

    tresh = -1 * sol[-len(vector):]
    
    return weights, tresh

weights, tresh = hopfield_perceptron(matrix)
np.savetxt('weights.txt', weights)
np.savetxt('tresh.txt', tresh)

hp = Hopfield.with_tresh(weights, tresh)
pprint(hp.list_attractors())

#plt.close('all')
#for state in hp.list_attractors():
#    hp.plot_neuron_states(4, 2, state)



    
    


