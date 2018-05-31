#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:17:32 2018

@author: hielke
"""

from custom_data_structs import Dataset, FeatureVector
#from Perceptron import Perceptron

from os import path
from typing import List

from matplotlib import pyplot as plt
from itertools import cycle, groupby

#from NearestNeighbour import NearestNeighbour


#def plotPerceptron(ds: Dataset, pc: Perceptron, 
#                   top: str, bottom: str, close_all: bool=True):
#    if close_all: 
#        plt.close('all')
#    plt.figure()
#    
#    plt.title("Perceptron")
#    plt.xlabel("Weight")
#    plt.ylabel("Size")
#    top_x, top_y, bottom_x, bottom_y = [[] for _ in range(4)]
#    for fv in ds:
#        if fv.label == 1:
#            x, y, _ = fv
#            top_x.append(x)
#            top_y.append(y)
#        else:
#            x, y, _ = fv
#            bottom_x.append(x)
#            bottom_y.append(y)
#    
#    plt.scatter(top_x, top_y, label=top, c='r', s=10)
#    plt.scatter(bottom_x, bottom_y, label=bottom, c='g', s=10)
#    
#    axes = plt.axis()
#    
#    x1, x2 = -100, 100
#    y1, y2 = map(lambda x: 
#        (-pc.weights[0] * x + pc.weights[2]) / pc.weights[1], 
#        [x1, x2])
#    plt.plot([x1, x2], [y1, y2], label="Decision boundary", c="k")
#    
#    x1, y1, _ = pc.weights
#    plt.plot([-100 * x1, 100 * x1], [-100 * y1, 100 * y1], label="Weights", c="y")
#    
#    # Update axes to situation before adding the two lines prevent 
#    # overextension of figure 
#    plt.xlim(axes[:2])
#    plt.ylim(axes[2:])
#    
#    plt.legend()
#    
#    plt.gca().set_aspect('equal', adjustable='box') # square box
#    
#    plt.show()
    
def digitFrame(title: str, image: List[float], 
               height: int, width: int, close_all: bool=True, tresh: bool=False):
    if tresh:
        image = image[:-1]
    
    assert height * width == len(image)
    
    if close_all:
        plt.close('all')
    plt.figure()
    
    plt.title(title)
    
    new_image = []
    for h in range(height):
        new_image.append(image[h*width:(h+1)*width])
    plt.imshow(new_image, cmap='gray')

#def plotNearestNeighbours(ds: Dataset, insert: FeatureVector, close_all: bool=True):
#    
#    if close_all: 
#        plt.close('all')
#    fig = plt.figure()
#    
#    
#    colors = cycle(['b', 'g', 'c', 'm', 'y', 'k', 'w'])
#    labelcolors = {label: next(colors) for label in ds.labels}
#    
#        
##    for fv in ds:
##        for index, label in enumerate(labels):
##            if fv.getLabel() == label:
##                x, y = fv
##                groups[index].append([x, y])
##    
#    plt.title("NearestNeighbours")
#    plt.xlabel("xlabel")
#    plt.ylabel("ylabel")
#    
##    for index,group in enumerate(groups):
##        x = [el[0] for el in group]
##        y = [el[0] for el in group]
##        color = labelcolors[index]
##        label = labels[index]
##        plt.scatter(x,y,label = label, c=color, s=10)
#    
#    vectors = sorted(ds, key=lambda f: f.label)
#    for label, fvs in groupby(vectors, lambda f: f.label):
#        plt.scatter(*zip(*fvs), label=label, c=labelcolors[label], s=10)
#    
#    plt.scatter(insert[0],insert[1],label = 'test ' + str(insert.getLabel()), c='k', s=10)
#    
#    plt.legend()
#    
#    nn = NearestNeighbour(ds)
#    
#    def onclick(event):
#        print('activated')
#        label = nn.predict([event.xdata, event.ydata], 3).label
#        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f, label=%d' %
#          (event.button, event.x, event.y, event.xdata, event.ydata, label))
#        
#        plt.plot(event.xdata, event.ydata, labelcolors[label] + '.')
#        fig.canvas.draw()
#
#    cid = fig.canvas.mpl_connect('button_press_event', onclick)
#    
#    # plt.gca().set_aspect('equal', adjustable='box') # square box
#    
#    plt.show()
#    
#if __name__ == '__main__':
#    a = "-1 0 16 96 240 192 16 0 0 0 112 255 96 96 160 0 0 0 128 255 32 0 176 32 0 0 80 255 48 0 80 112 0 0 112 208 48 0 128 112 0 0 64 192 0 16 208 80 0 0 0 224 144 240 144 0 0 0 0 96 224 112 16 0 0"
#    a = a.split()
#    a = a[1:]
#    a = list(map(float, a))
#    digitFrame('Test', a, 8, 8)
#    
#    ds = Dataset(path.join('data', 'test_file.txt'), addTresh=True)
#    pc = Perceptron(1)
#    pc.updateWeights(ds)
#    plotPerceptron(ds, pc, 'apples', 'pears', close_all=False)