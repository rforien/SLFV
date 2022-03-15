#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:15:57 2018

@author: rforien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Labelled_Coalescent(object):
    def __init__(self, n, labels):
        assert type(n) == int, "Unexpected type of sample size."
        assert n >= 1, "Sample size must be at least 1."
        assert type(labels) == np.ndarray, "Unexpected type of labels."
        assert np.size(labels, 0) == n, "Wrong number of labels provided."
        self.coalescent = np.reshape(np.arange(n), (1,n))
        self.labels = np.reshape(labels, (1,) + np.shape(labels))
    
    def single_merger(self, indices_to_merge, new_label):
        merge = [i in indices_to_merge for i in self.coalescent[-1,]]
        if any(merge):
            self.coalescent = np.vstack((self.coalescent, self.coalescent[-1,]))
            self.labels = np.vstack((self.labels, np.reshape(self.labels[-1,], (1,) + np.shape(self.labels[-1,]))))
            self.labels[-1,merge,] = new_label
            self.coalescent[-1,merge] = np.min(indices_to_merge)
    
    def kill(self, index):
        self.single_merger([index], np.reshape(np.nan*np.ones(np.size(self.labels[-1,0])), (1, np.size(self.labels[-1,0]))))
    
    def alive_lineages(self):
        return [x for x in self.coalescent[-1,:] if self.current_labels()[x,0] < np.inf]
    
    def nb_alive_lineages(self):
        return np.size(self.alive_lineages())
    
    def current_labels(self):
        return self.labels[-1,]
    
    def current_coalescent(self):
        return self.coalescent[-1,]
    
    def nb_current_lineages(self):
        return np.size(np.unique(self.coalescent[-1,]))

class SLFV_dual(object):
    number_types = (int, float, np.int64, np.float64)
    def _set_r(self, r):
        assert type(r) in self.number_types, "Unexpected type for radius of reproduction events."
        assert r > 0, "Radius of reproduction events must be positive."
        self._r = float(r)
    def _get_r(self):
        return self._r
    r = property(_get_r, _set_r)
    
    def _set_L(self, L):
        assert type(L) in self.number_types, "Unexpected type for L."
        assert L > 0, "L must be positive."
        self._L = float(L)
    def _get_L(self):
        return self._L
    L = property(_get_L, _set_L)
    
    def _set_u(self, u):
        assert type(u) in self.number_types, "Unexpected type for impact parameter."
        assert u > 0 and u <= 1, "Impact parameter must be in (0,1]."
        self._u = float(u)
    def _get_u(self):
        return self._u
    u = property(_get_u, _set_u)
    
    def _set_running_time(self, t):
        assert type(t) in self.number_types, "Unexpected type of running time."
        assert t >= 0, "Running time must be non-negative."
        self._running_time = float(t)
    def _get_running_time(self):
        return self._running_time
    running_time = property(_get_running_time, _set_running_time)
    
    def _set_killing_rate(self, k):
        assert type(k) in self.number_types, "Unexpected type of killing rate."
        assert k >= 0, "Killing rate must be non-negative."
        self._killing_rate = float(k)
    def _get_killing_rate(self):
        return self._killing_rate
    killing_rate = property(_get_killing_rate, _set_killing_rate)
    
    def __init__(self, L = 10, r = 1, u = .1, d = 1, mutation_rate = 0.):
        self.L = L
        self.r = r
        self.u = u
        assert d in (1, 2, 3), "Dimension must be either 1, 2 or 3."
        self.d = d
        self.killing_rate = mutation_rate
        
        self.init_lineages_uniform(1)
        
    def init_lineages_uniform(self, n = 1):
        assert type(n) == int, "Unexpected type for number of lineages to initiate."
        assert n >= 1, "There must be at least one lineage."
        self.running_time = 0.
        self.coalescence_times = [0.]
        self.coalescent = Labelled_Coalescent(n, self.L*np.random.random((n, self.d)))
    
    def init_lineages(self, lineages):
        assert type(lineages) == np.ndarray, "Unexpected type of initial lineages."
        assert np.min(lineages) >= -self.L/2 and np.max(lineages) <= self.L/2, "At least one lineage is outside the grid!"
        assert np.size(lineages) > 0, "Empty lineages."
        self.running_time = 0.
        self.coalescence_times = [0.]
        self.coalescent = Labelled_Coalescent(np.size(lineages, 0), lineages + self.L/2)
    
    def _total_rate_events(self, T):
        assert type(T) in self.number_types, "Unexpected type for T."
        assert T >= 0, "T must be non-negative."
        return self.L**self.d*float(T)
    
    def V(self, r):
        assert self.d in (1,2,3), "Ouups! Dimension should be either 1 or 2, volume of ball not implemented otherwise."
        if self.d == 1:
            return 2.*r
        elif self.d == 2:
            return np.pi*r**2
        elif self.d == 3:
            return (4./3.)*np.pi*r**3
    
    def _reproduction_event(self, x = None, r = None):
        ''' returns the genealogical 'map' from the previous generation to the next '''
        if type(x) == type(None):
            x = self.L*np.random.random(self.d)
        else:
            assert type(x) == np.ndarray, "Unexpected type for center of reproduction event."
            assert np.shape(x) == (self.d, ), "Wrong shape of center of reproduction event."
        if type(r) == type(None):
            r = self.r
        else:
            assert type(r) in self.number_types, "Unexpected type of radius of reproduction event."
            assert r > 0, "Radius of reproduction event must be positive."
        lineages_in_event = np.sqrt(np.sum((x - self.coalescent.current_labels())**2, 1)) <= r
        lineages_in_event = lineages_in_event*(np.random.random(np.size(lineages_in_event)) < self.u)
        involved_lineages = [i for i in self.coalescent.current_coalescent() if lineages_in_event[i]]
        if np.size(involved_lineages) == 0:
            return
        lower = np.maximum(np.zeros(self.d), x - r)
        upper = np.minimum(self.L*np.ones(self.d), x + r)
        y = lower + (upper-lower)*np.random.random(self.d)
        self.coalescent.single_merger(involved_lineages, np.reshape(y, (1, self.d)))
        self.coalescence_times = np.concatenate((self.coalescence_times, [self.running_time]))
    
    def run(self, T, verbose = False):
        number_events = np.random.poisson(self._total_rate_events(T))
        t = self.running_time
        times = float(T)*np.random.random(number_events)
        times = self.running_time + np.sort(times)
        for i in np.arange(number_events):
            self.running_time = times[i]
            self._reproduction_event()
            if verbose:
                print("\rProgress: %d%%.  " % (100*float(i)/number_events),)
        if verbose:
            print('')
        self.running_time = t + T
        
    def _pick_around(self, x, r):
        y = x + 2*r
        while np.sqrt(np.sum((x-y)**2)) > r:
            y = x - r + 2*r*np.random.random(np.shape(x))
        return y
    
    def run_efficient(self, T, verbose = False):
        assert type(T) in self.number_types, "Unexpected time for T."
        assert T > 0, "T must be positive."
        t = self.running_time
        while self.running_time < t + T and self.coalescent.nb_alive_lineages() > 0:
            rate = self.V(self.r)*self.coalescent.nb_alive_lineages()
            interval = np.random.exponential(1./rate)
            self.running_time += interval
            unique = np.unique(self.coalescent.alive_lineages())
            lineage = np.random.choice(unique)
            center = self._pick_around(self.coalescent.current_labels()[lineage], self.r)
            overlap = np.sum(np.sqrt(np.sum((self.coalescent.current_labels()[unique] - center)**2, 1)) <= self.r)
            if np.random.random() < 1./overlap:
                self._reproduction_event(x=center)
            if self.killing_rate > 0:
                self._killing(interval)
            if verbose:
                print("\rProgress: %d%%.  " % (100*float(self.running_time - t)/T),)
        self.running_time = t + T
        if verbose:
            print('')
    
    def _killing(self, t):
        assert type(t) in self.number_types, "Unexpected type for t."
        assert t >= 0, "t must be non-negative."
        unique = np.unique(self.coalescent.alive_lineages())
        killed = np.where(np.random.random(np.size(unique)) < 1. - np.exp(-self.killing_rate*t))[0]
        for (i, k) in enumerate(unique[killed]):
            self.coalescent.single_merger([k], np.reshape(np.nan*np.ones(self.d), (1, self.d)))
            self.coalescence_times = np.concatenate((self.coalescence_times, [self.running_time]))
            
    
    def display_trajectory(self, thickness = .5):
        assert self.d == 1, "This is only possible if d is 1."
        assert type(thickness) in self.number_types, "Unexpected type for thickness parameter."
        plt.figure()
        plt.xlim((np.nanmin(self.coalescent.labels) - self.L/2, np.nanmax(self.coalescent.labels) - self.L/2))
        unique, inverse = np.unique(self.coalescent.current_coalescent(), return_inverse = True)
        raw_colors = [cm.jet(x) for x in np.linspace(0,1, np.size(unique))]
        colors = [raw_colors[i] for i in inverse]
        times = np.concatenate((np.repeat(self.coalescence_times, 2)[1:], [self.running_time]))
        for i in np.arange(np.size(self.coalescent.labels, 1)):
            trajectory = np.repeat(self.coalescent.labels[:,i,0] - self.L/2., 2)
            plt.plot(trajectory, times, '-', linewidth = thickness, color = colors[i])
        plt.ylabel("Time (generations ago)", fontsize = 12)
        plt.xlabel("Space", fontsize = 12)

class SLFV_dual_heterogeneous_dispersal(SLFV_dual):
    def _set_r_left(self, r):
        assert type(r) in self.number_types, "Unexpected type of radius of reproduction event."
        assert r > 0, "Radius of reproduction events must be positive."
        self._r_left = float(r)
    def _get_r_left(self):
        return self._r_left
    r_left = property(_get_r_left, _set_r_left)
    
    def _set_r_right(self, r):
        assert type(r) in self.number_types, "Unexpected type of radius of reproduction event."
        assert r > 0, "Radius of reproduction events must be positive."
        self._r_right = float(r)
    def _get_r_right(self):
        return self._r_right
    r_right = property(_get_r_right, _set_r_right)
    
    def __init__(self, r_left = 1, r_right = 1, *args, **kwargs):
        SLFV_dual.__init__(self, *args, **kwargs)
        self.r_right = r_right
        self.r_left = r_left
    
    def _total_rate_events(self, T):
        return (0.5/self.V(self.r_left)+0.5/self.V(self.r_right))*SLFV_dual._total_rate_events(self, T)
    
    def _reproduction_event(self, x = None, r = None):
        if type(x) == type(None):
            scale = np.ones(self.d)
            scale[0] = 0.5
            x = scale * self.L * np.random.rand(self.d)
            if np.random.rand() < 1./(1.+(self.r_right/self.r_left)**self.d):
                x[0] = 0.5*self.L + x[0]
        if r == None:
            if x[0] < 0.5 * self.L:
                r = self.r_left
            else:
                r = self.r_right
        SLFV_dual._reproduction_event(self, x, r)
    
    def display_trajectory(self, *args, **kwargs):
        SLFV_dual.display_trajectory(self, *args, **kwargs)
        plt.vlines([0], 0, self.running_time)
        
            
if __name__ == "__main__":
    ##dual = SLFV_dual_heterogeneous_dispersal(L = 200, r_left = .8, r_right = 2, u = .3)
    dual = SLFV_dual(L = 100, r = 1, u = .2, mutation_rate = .001, d=1)
    dual.init_lineages_uniform(20)
    dual.run_efficient(400, verbose = True)
    dual.display_trajectory(thickness = .5)