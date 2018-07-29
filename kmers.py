#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 22:53:53 2018

@author: hielke
"""

from typing import Iterable
from itertools import tee, repeat

A = "AGCTTTTCATTCTGACT"
B = "GGTAATGTGGTGACCAT"
C = "CCTTTGCAGTCCAGCAC"

def k_tuple(iterable: Iterable, k: int) -> Iterable:
    """Creates tuples of length k from the given iterator.
    
    Parameters
    ----
    it : iter
        The iterable from which the k tuples are created
    k : int
        The lenght of the k tuple
        
    Returns
    ----
    iter
        Returns the result as an iterator which generates the tuples.
    
    Example
    ----
    >>> k_tup = k_tuple('abcd', 2)
    >>> next(k_tup)
    ('a', 'b')
    >>> list(k_tup)
    [('b', 'c'), ('c', 'd')]
    """
    iters = tee(iterable, k)
    for ind, it in enumerate(iters):
        for _ in repeat(None, ind):
            next(it)
    return zip(*iters)

#kmers = []
#for ss in A, B, C:
#    kmers.extend(list(k_tuple(ss, 5)))
#    
#print(kmers)
reads = ["CTTTAGATT", 
         "TGCAGTCGA", 
         "TCCAGGACT",
         "AGTCCAGCA", 
         "TTTGCAGTC"]
k = 5
for ind, read in enumerate(reads):
    print("examening read %d" % ind)
    for abc, ss in enumerate((A, B, C)):
        ss = k_tuple(ss, k)
        count = 0
        for r in k_tuple(read, k):
            if r in ss:
                count += 1
        
        print("%s has count %d" % (chr(abc + 97), count))
        