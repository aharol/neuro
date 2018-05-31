#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:18:34 2018

@author: hielke
"""

import numpy as np
from random import sample
from itertools import repeat, chain


inp_good = []
inp_good.append(np.array([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]))
inp_good.append(np.array([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]))
#weights_good = np.array([[0, 1, 0],
 #                   [0, 1, 0],
  #                  [0, 1, 0]])
inp_bad = []
inp_bad.append(np.array([[1, 1, 1],
                             [1, 0, 1],
                             [0, 0, 1]]))
    
inp_bad.append(np.array([[1, 0, 1],
                             [0, 1, 1],
                             [1, 0, 1]]))

inp_bad.append(np.array([[0, 0, 1],
                             [1, 0, 0],
                             [1, 0, 1]]))

inp_bad.append(np.array([[0, 1, 1],
                             [1, 1, 0],
                             [1, 1, 1]]))

inp_bad.append(np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]]))

inp_bad.append(np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]]))

#inp_bad.append(np.array([[0, 0, 0],
#                             [0, 1, 1],
#                             [0, 1, 0]]))
#inp_bad.append(np.array([[0, 1, 0],
#                             [0, 1, 1],
#                             [0, 0, 0]]))
#inp_bad.append(np.array([[0, 1, 0],
#                             [1, 1, 0],
#                             [0, 0, 0]]))
#inp_bad.append(np.array([[0, 0, 0],
#                             [1, 1, 0],
#                             [0, 1, 0]]))


#inp_good = []
#inp_good.append(np.array([[1, 0, 0],
#                          [1, 0, 0],
#                          [1, 0, 0]]))
#
#inp_good.append(np.array([[0, 0, 1],
#                          [0, 0, 1],
#                          [0, 0, 1]]))
#
#inp_bad = []
#inp_bad.append(np.array([[1, 0, 1],
#                         [1, 0, 1],
#                         [1, 0, 1]]))

class perceptron:

    def __init__(self, inp_bad, inp_good):    
        self.weights = np.ones((3,3))
        self.rate = 0.01
        self.treshold = 1
        self.inp_bad = inp_bad
        self.inp_good = inp_good
        
    def test(self, inp):
        return 1 if self.treshold <= sum(np.nditer(inp * self.weights)) else -1

    def train(self, iters):
        
        for _ in repeat(None, iters):
            still_wrong = False
            for tester in chain(zip(inp_good, repeat(1)), 
                                    zip(inp_bad, repeat(-1))):
                if self.test(tester[0]) != tester[1]: # wrong classified: update
                    self.weights += self.rate * tester[1] * tester[0]
                    self.treshold -= self.rate * tester[1] * self.treshold
                    still_wrong = True
            if not still_wrong:
                print("We are good")
                break
                    
p = perceptron(inp_bad, inp_good)
p.train(1000)
            
        

        
        
        
    







