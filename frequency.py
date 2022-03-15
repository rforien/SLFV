#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:30:50 2022

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Frequency1D(object):
    def __init__(self, shape, dx, nb_alleles, freq = None):
        assert np.min(shape) > 0
        assert type(nb_alleles) == int and nb_alleles >= 2
        assert dx > 0
        self.shape = shape
        self.dx = dx
        self.N = nb_alleles
        self._build_space_freq(freq)
    
    def _build_space_freq(self, freq):
        self.nx = int(self.shape/self.dx)+1
        self.space = np.linspace(0, self.shape, self.nx)
        if type(freq) != type(None):
            if np.shape(freq) == (self.nx, self.N):
                self.freq = freq
            else:
                raise Exception("Shape mismatch.")
        else:
            self.freq = np.ones((self.nx, self.N))/self.N
        self.colors = [cm.jet(x) for x in np.linspace(0,1,self.N)]
    
    def freq_in_ball(self, centre, radius):
        mean = np.mean(self.freq[np.abs(self.space-centre) <= radius], axis = 0)
        return mean
    
    def update(self, centre, radius, impact, new_allele):
        ball = np.abs(self.space-centre) <= radius
        self.freq[ball] = (1-impact)*self.freq[ball] + impact*np.array(np.arange(self.N) == new_allele)
        
    def set_freq(self, freq_func, *args, **kwargs):
        try:
            self.freq = freq_func(self.space, *args, **kwargs)
        except:
            for (i, z) in enumerate(self.space):
                self.freq[i] = freq_func(z, *args, **kwargs)
        
    def plot(self, ax = None, fill = True):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        summed_freq = np.cumsum(self.freq, axis = 1)
        lines = tuple()
        for i in range(self.N):
            if fill:
                if i==0:
                    ax.fill_between(self.space, 0, summed_freq[:,i], color=self.colors[i])
                else:
                    ax.fill_between(self.space, summed_freq[:,i-1], summed_freq[:,i], color = self.colors[i])
            else:
                _line, = ax.plot(self.space, summed_freq[:,i])
                lines = lines + (_line,)
        ax.set_ylim((0,1))
        if self.N == 2:
            ax.set_ylabel("Allele Frequency")
        else:
            ax.set_ylabel("Cumulative allele frequencies")
        ax.set_xlabel("Space")
        return lines
    
    def update_plot(self, ax, lines, fill = True):
        summed_freq = np.cumsum(self.freq, axis = 1)
        if not fill:
            for i in range(self.N):
                lines[i].set_data(self.space, summed_freq[:,i])
            return lines
        else:
            for artist in plt.gca().lines + plt.gca().collections:
                artist.remove()
            ax.fill_between(self.space, 0, summed_freq[:,0], color=self.colors[0])
            for i in range(1,self.N):
                ax.fill_between(self.space, summed_freq[:,i-1], summed_freq[:,i], color=self.colors[i])
            return ax
    
    def __add__(self, other):
        if isinstance(other, Frequency1D):
            new_freq = self.freq + other.freq
        else:
            new_freq = self.freq + other
        return Frequency1D(self.shape, self.dx, self.N, freq = new_freq)                
    
    def __mul__(self, cst):
        new_freq = cst * self.freq
        return Frequency1D(self.shape, self.dx, self.N, freq = new_freq)
    
    def __rmul__(self, cst):
        return self.__mul__(cst)

class Frequency2D(Frequency1D):
    def _build_space_freq(self, freq):
        self.nx = tuple(np.floor(np.array(self.shape)/self.dx).astype(int))
        x = np.linspace(0, self.shape[0], self.nx[0])
        y = np.linspace(0, self.shape[1], self.nx[1])
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        if type(freq) != type(None):
            if np.shape(freq) == self.nx + (self.N,):
                self.freq = freq
            else:
                raise Exception("Shape mismatch.")
        else:
            self.freq = np.ones(self.nx + (self.N,))/self.N
    
    def freq_in_ball(self, centre, radius):
        ball = np.sqrt((self.X - centre[0])**2 + (self.Y-centre[1])**2) <= radius
        return np.mean(self.freq[ball], axis = 0)
    
    def update(self, centre, radius, impact, new_allele):
        ball = np.sqrt((self.X - centre[0])**2 + (self.Y-centre[1])**2) <= radius
        self.freq[ball] = (1-impact) * self.freq[ball] + impact * np.array(np.arange(self.N) == new_allele)
    
    def set_freq(self, freq_func, *args, **kwargs):
        try:
            self.freq = freq_func(self.X, self.Y, *args, **kwargs)
        except Exception as e:
            print(e)
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    self.freq[i,j] = freq_func(self.X[i,j], self.Y[i,j], *args, **kwargs)
    
    def plot(self, ax = None, colorbar = True):
        if ax is None:
            plt.figure()
            ax = plt.axes()
            colorbar = True
        else:
            ax.clear()
        if self.N == 2:
            Z = self.freq[:,:,0]
            vmax = 1
            levels = np.linspace(0, 1, 101, endpoint = True)
        else:
            Z = np.argmax(self.freq, axis = 2)+.1
            vmax = self.N
            levels = np.arange(self.N+1)
        lines = ax.contourf(self.X, self.Y, Z, vmin=0, vmax=vmax, levels = levels, cmap = cm.jet)
        if colorbar:
            cb = plt.colorbar(lines)
            if self.N == 2:
                cb.set_label("Proportion of allele 0")
            else:
                cb.set_label("Majority allele")
        plt.show()
        return ax
    
    def update_plot(self, ax, lines, fill = None):
        return self.plot(ax, colorbar=False)