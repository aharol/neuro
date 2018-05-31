#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:17:49 2018

@author: hielke
"""

from collections import UserList
from typing import List
from math import sqrt

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

class FeatureVector(UserList): 
    """A subclass of a list with some additional features usefull for a feature 
    vector.
    Can be used just like a list. 
    """
    
    def __init__(self, label: int):
        super().__init__()
        #self.data = [] #UserList.__init__(self)
        self.label = label
    
    def product(self, weights: List[float]):
        ans = 0.0
        for s, w in zip(self, weights):
            ans += s * w
        return ans
    
    def distance(self, vector: List[float]):
        assert len(self) == len(vector)
        ans = 0.0
        for s, v in zip(self, vector): # !! waarom skipt hij de eerste entry van deze iterator? Waarom heeft hij niet gewoon als eerste waarde (1,2) en als tweede ([1,2],[3,4]) 
            ans += (s - v) ** 2        # -> Dit wordt wel als goed getest,
        return sqrt(ans)
    
    def __str__(self):
        return "<%d, %s>" % (self.label, str(list(map(prettyfloat, self))))
    
    def getLabel(self):
        return self.label
    
class Dataset(UserList):
    
    def __init__(self, filename: str=None, addTresh: bool=None):
        super().__init__() # type: List[FeatureVector]
        self.featureSize = -1
        self.labels = set()
        if filename:
            self.readData(filename, addTresh)
            
    def readData(self, filename: str, addTresh: bool):
        with open(filename) as f:
            first_line = next(f)
            self.featureSize = len(first_line.split()) - 1 
            self.addFeatureVector(first_line, addTresh)
            for line in f:
                self.addFeatureVector(line, addTresh)
            
    def addFeatureVector(self, line: str, addTresh: bool):
        parts = line.split()
        assert len(parts) - 1 == self.featureSize
        parts = iter(parts)
        label = int(next(parts))
        self.labels.add(label)
        fv = FeatureVector(label=label)
        fv.extend(map(float, parts))
        if addTresh:
            fv.append(-1.0) # !! waarom append hij de list in de list?
        self.append(fv)

if __name__ == '__main__':
    fv = FeatureVector(0)
    print(fv)
    fv = FeatureVector(1)
    fv.extend([1,2])
    print(fv)