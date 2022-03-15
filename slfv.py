#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:06:03 2022

@author: raphael
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import frequency
import event_drawer as events
import time

class SLFV(object):
    def __init__(self, shape, nb_alleles, event_dist, dx = 0.1):
        assert dx > 0
        assert type(nb_alleles) == int and nb_alleles >= 2
        assert isinstance(event_dist, events.EventDist)
        assert np.min(shape) > 0
        self.d = np.size(shape)
        if self.d == 1:
            self.frequency = frequency.Frequency1D(shape, dx, nb_alleles)
        elif self.d == 2:
            self.frequency = frequency.Frequency2D(shape, dx, nb_alleles)
        else:
            raise Exception("Dimension other than 1 or 2 not implemented.")
        self.event_dist = event_dist
        
    def set_freq(self, freq_func, *args, **kwargs):
        self.frequency.set_freq(freq_func, *args, **kwargs)
        
    def plot(self, ax = None, show = True):
        if ax is None:
            self.fig = plt.figure()
            self.ax = plt.axes()
        else:
            self.ax = ax
        self.lines = self.frequency.plot(self.ax, show = show)
        if ax is None:
            plt.tight_layout()
    
    def run(self, T, verbose = False, interactive = False):
        rate = self.event_dist.intensity(self.frequency.shape)
        number_of_events = rand.poisson(rate * T)
        # times = rand.rand(number_of_events) * T
        # times = np.concatenate(([0], times, [T]))
        if (interactive):
            self.plot()
        for i in range(number_of_events):
            # wait for click if interactive
            if (interactive):
                self.plot(self.ax)
                plt.draw()
                plt.waitforbuttonpress()
            else:
                time.sleep(0.001)
            if (verbose):
                print("\rProgress: %d%%.  " % (100*float(i)/number_of_events))
            self.event_dist.run_event(self.frequency)
        if verbose:
            print('')
    
    def run_animate(self, T, dt = 0.1, save = False, name = '', fill = True):
        assert dt > 0
        self.plot(show = False)
        anim = animation.FuncAnimation(self.fig, self._update, interval = 80, blit=False, fargs=(dt, fill))
        if (save and name != ''):
            anim.save(name, writer ='ffmpeg')
        else:
            plt.show()
    
    def _update(self, i, dt, fill = True):
        self.run(dt)
        time.sleep(0.03)
        result = self.frequency.update_plot(self.ax, self.lines, fill)
        return result
        
if __name__ == '__main__':
    # u = .2
    # r = .5
    # alpha = 1.5
    # event = events.FixedEvent(u, r)
    # event = events.StableRadius(u, u, alpha, 2)
    
    u0 = .3
    r0 = 1
    c = 0
    b = 5
    a = 1.5
    mu = 0
    event = events.OneTailRadiiWithMutation(u0, r0, c, b, a, 2, mu)
    
    slfv = SLFV((10, 10), 2, event, dx = 0.05)
    def freq_func(X, Y, X0, Y0, R):
        freq = np.zeros(slfv.frequency.nx + (slfv.frequency.N,))
        freq[:,:,0] = (np.sqrt((X-X0)**2+(Y-Y0)**2)<R)
        freq[:,:,1] = 1-freq[:,:,0]
        return freq
    X0 = 5
    Y0 = 1
    R = 4
    slfv.set_freq(freq_func, X0=X0, Y0=Y0, R=R)
    slfv.run_animate(3, dt = 0.02)