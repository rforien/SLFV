#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:03:35 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt

class PiecewiseConstant(object):
    def __init__(self, x, y):
        assert np.size(x) == np.size(y)
        x = np.array(x)
        x, index, inverse, counts = np.unique(x, return_index= True, 
                                              return_counts= True,
                                              return_inverse = True)
        y_ = np.array(y)[index]
        if np.max(counts) > 1:
            if not (y_[inverse] == y).all():
                raise Exception("Ambiguous definition.")
        y = y_
        # merge successive intervals with identical y values
        index = np.hstack(([True], np.diff(y)!=0))
        self.x = x[index]
        self.y = y[index]
    
    def __call__(self, x):
        try:
            indexes = np.array([np.sum(self.x <= xi)-1 for xi in x])
        except TypeError:
            indexes = np.sum(self.x <= x)-1
        if np.min(indexes) < 0:
            raise Exception("Trying to evaluate a function outside of its domain.")
        return self.y[indexes]
    
    def __add__(self, g):
        try:
            x = np.hstack((self.x, g.x))
            y = np.hstack((self.y + g(self.x), self(g.x) + g.y))
            return PiecewiseConstant(x, y)
        except:
            y = self.y + g
            return PiecewiseConstant(self.x, y)
    
    def __neg__(self):
        return PiecewiseConstant(self.x, -self.y)
    
    def __sub__(self, g):
        return self + (-g)
    
    def __mul__(self, g):
        if isinstance(g, PiecewiseConstant):
            x = np.hstack((self.x, g.x))
            y = np.hstack((self.y * g(self.x), self(g.x) * g.y))
            return PiecewiseConstant(x, y)
        else:
            y = self.y * g
            return PiecewiseConstant(self.x, y)
    
    def maximum(self):
        return np.max(self.y)
    
    def plot(self, ax = None, xlim = None, *args, **kwargs):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        if xlim is None:
            xlim = (np.min(self.x), np.max(self.x))
        if xlim[0] < np.min(self.x):
            raise Exception("Plot interval outside of definition domain.")
        x_min, x_max = xlim
        x, y = self.x[self.x > x_min], self.y[self.x > x_min]
        x, y = x[x < x_max], y[x < x_max]
        i = np.sum(self.x <= x_min) - 1
        j = np.sum(self.x <= x_max) - 1
        x, y = np.hstack(([x_min], x, [x_max])), np.hstack(([self.y[i]], y, [self.y[j]]))
        x = np.repeat(x, 2)[1:]
        y = np.repeat(y, 2)[:-1]
        ax.plot(x, y, *args, **kwargs)
        return ax
        
            