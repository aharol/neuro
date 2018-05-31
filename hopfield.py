#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:12:40 2018

@author: hielke
"""

import numpy as np
import random
from pprint import PrettyPrinter as pp
pprint = pp().pprint

class Hopfield:
    
    def __init__(self, weights):
        weights = np.asarray(weights)
        size = len(weights)
        assert size * size == np.size(weights), \
                    "Weights should be square matrix"

        self.weights = weights
        self.size = size
        
        self.reset_states()
        self.tresholds = np.zeros(size) 
        
        assert not (self.weights * np.eye(self.size)).all(), \
                "There are self-loops"
        assert np.allclose(self.weights, self.weights.T), \
                "Not symmetric"
        
    @classmethod
    def hebbian(cls, vector):
        size = np.size(vector)
        vector = np.asmatrix(vector)
        weights = vector.T * vector - np.eye(size)
        return cls(weights)
    
    @classmethod
    def multiple_hebbian(cls, matrix):
        size = np.size(matrix[0, :])
        amm_vectors = len(matrix)
        assert np.size(matrix) == size * amm_vectors, \
                "Not all vectors are of the same length"
        weights = np.matlib.zeros((size, size))
        for i in range(amm_vectors):
            vec = np.asmatrix(matrix[i, :])
            weights += vec.T * vec
        weights -= amm_vectors * np.matlib.eye(size)
        return cls(weights)
    
    @classmethod
    def with_tresh(cls, weights, tresh):
        hp = cls(weights)
        hp.tresholds = tresh
        return hp
        
    def reset_states(self):
        self.states = np.array([random.choice([-1, 1]) 
                for _ in range(self.size)], dtype=int)
    
    def update(self):
        i = random.choice(range(self.size))
        
        delta = 0
        for j in range(self.size):
            delta += self.weights[i, j] * self.states[j]
        
        new_state = 1 if delta >= self.tresholds[i] else -1
        
        changed = self.states[i] != new_state
        self.states[i] = new_state
        
        return i, changed
        
    def converge(self, states=None):
        if states: self.states=states
        changed_states = [True for _ in range(self.size)]
        counter = 0
        while any(changed_states):
            i, change = self.update()
            if change:
                changed_states = [True for _ in range(self.size)]
            else:
                changed_states[i] = False
            counter += 1
            if counter > 10000:
                return print("ERR: exceeded maximum iterations.")
        return self.states
    
    def list_attractors(self):
        attr_list = []
        
        for _ in range(1000):
            self.reset_states()
            attr = self.converge()
            
            for a in attr_list:
                if np.array_equal(a, attr):
                    break
            else:
                attr_list.append(attr)
        
        return attr_list
    
    def plot_neuron_states(self, height, width, states=None):
        if states is None:
            states = self.states
        print(height, width, np.size(states))
        assert height * width == np.size(states), \
                "Height and width do not match the size of the states."
        
        image = []
        
        for h in range(height):
            image.append(states[h*width:(h+1)*width])
#        plt.figure()
#        plt.imshow(image, cmap='gray')
        

def simple_convergence():
    weights = np.matrix([[0, -1, 1],
                        [0, 0, -1],
                        [0, 0, 0]])
    weights += weights.T
    hp = Hopfield(weights)
    print("attractors:")
    pprint(hp.list_attractors())
    
    
def simple_hebbian():
    vector = np.matrix([1, -1, 1, -1, 1, 1, 1, 1])
    hp = Hopfield.hebbian(vector)
    print("attractors:")
    pprint(hp.list_attractors())
    
def simple_hebbian2():
    vector = np.matrix([1,1,1,1])
    hp = Hopfield.hebbian(vector)
    print("attractors:")
    pprint(hp.list_attractors())
    
def multiple_hebbian():
    matrix = np.array([[1, 1, 1, 1],
                               [1, -1, 1, -1],
                               [1, 1, -1, -1]])
    hp = Hopfield.multiple_hebbian(matrix)
    print("attractors:")
    pprint(hp.list_attractors())
    print("current state:")
    
def test2():
    matrix = np.array([[1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, -1]])
    hp = Hopfield.multiple_hebbian(matrix)
    print("attractors:")
    pprint(hp.list_attractors())

def with_an_image():
    matrix = np.array([[1, 1, -1, 1, -1, 1, -1, -1],
                       [1, 1, -1, 1, -1, -1, -1, 1],
                       [1, 1, 1, 1, -1, -1, -1, -1],
                       [-1, 1, -1, 1, -1, 1, -1, 1]])


    hp = Hopfield.multiple_hebbian(matrix)
    print("attractors:")
    pprint(hp.list_attractors())
    print("current state:")
    print(hp.converge())
    hp.plot_neuron_states(4, 2)

if __name__ == '__main__':
#    sim1ple_convergence()
    
#    simple_hebbian()
    
#    simple_hebbian2()
    
#    with_an_image()
    
    test2()
    













