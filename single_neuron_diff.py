#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:19:39 2018

@author: hielke
"""

import numpy as np
from matplotlib import pyplot as plt

t_start = 0
t_max = 1000
t_step = 0.01
V_rest = 0 # start
V_reset = 0 # relaxation
V_theta = 0.5 # treshold var
V_exi = 1 # exitation
C = 1 # μF
tau = 100
I = 1.5
R = 1
theta = 0.5


t = np.arange(t_start, t_max, t_step)
steps = len(t)
V = np.zeros(steps)
V[0] = V_rest
for ii in range(steps)[1:]:
    if V[ii-1] == V_exi: # relaxation
        V[ii] = V_reset
        continue
    
    V_step = (-V[ii-1] + I * R)/tau
    V[ii] = V[ii-1] + t_step * V_step
    
    if V[ii] > V_theta: # exitation
        V[ii] = V_exi
        continue
    



plt.plot(t, V, label="Electronic potential")
plt.legend()
plt.show()