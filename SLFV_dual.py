#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:15:57 2018

@author: rforien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from . import events
from . import ARG

class Labelled_Coalescent(object):
    def __init__(self, labels):
        assert type(labels) == np.ndarray, "Unexpected type of labels."
        n = np.size(labels, axis = 0)
        self.coalescent = np.reshape(np.arange(n), (1,n))
        self.labels = np.reshape(labels, (1,) + np.shape(labels))
        self.times = [0]
    
    def single_merger(self, time, indices_to_merge, new_label):
        merge = [i in indices_to_merge for i in self.coalescent[-1,]]
        if any(merge):
            self.coalescent = np.vstack((self.coalescent, self.coalescent[-1,]))
            self.labels = np.vstack((self.labels, np.reshape(self.labels[-1,], (1,) + np.shape(self.labels[-1,]))))
            self.labels[-1,merge,] = new_label
            self.coalescent[-1,merge] = np.min(indices_to_merge)
        self.times.append(time)
    
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
    
def dist(positions, centre):
    return np.sqrt(np.sum((positions - centre[np.newaxis,:])**2, axis = 1))

class SLFV_dual(object):
    def __init__(self, event_dist, dim):
        assert type(dim) == int and dim >= 1
        self.d = dim
        self.event_dist = event_dist
    
    def run_coalescent(self, lineages_init_positions, T, verbose = False):
        assert np.size(lineages_init_positions, axis = 1) == self.d
        self.n = np.size(lineages_init_positions, axis = 0)
        self.init_coalescent(lineages_init_positions)
        self.times = [0]
        t = 0
        while True:
            if verbose:
                print("Progress: %.1f%%" % (100*(t / T)))
            positions = self.get_current_positions()
            # draw the time of next event
            rates = self.event_dist.jump_rates(positions)
            dt = np.random.exponential(1/np.sum(rates))
            # stop if we've reached T
            if t + dt > T:
                t = T
                break
            else:
                t = t + dt
            # draw the lineage involved in the event
            i = np.random.choice(np.arange(len(rates)), p = rates / np.sum(rates))
            try:
                params = self.event_dist.draw_event_params(positions[i])
            except events.IgnoreEvent:
                continue
            lineages_in_ball = dist(positions, params['centre']) <= params['radius']
            assert lineages_in_ball[i], "Involved lineage not in the ball"
            invovled_lineages = np.random.binomial(1, 
                                    p = lineages_in_ball * params['impact'])
            invovled_lineages[i] = 1
            # thinning to avoid multiple counting
            if np.sum(invovled_lineages) > 1:
                if np.random.uniform() > 1/np.sum(invovled_lineages):
                    if verbose:
                        print("Ignoring event")
                    continue
            # add merger
            indices_to_merge = np.arange(len(invovled_lineages))[invovled_lineages == True]
            self.merge(t, indices_to_merge, params, verbose = verbose)
            self.times.append(t)
        self.times = np.array(self.times)
    
    def init_coalescent(self, lineages_init_positions):
        self.coalescent = Labelled_Coalescent(lineages_init_positions)
    
    def get_current_positions(self):
        return self.coalescent.current_labels()
    
    def merge(self, time, indices_to_merge, event_params, verbose = False):
        self.coalescent.single_merger(time,
                                      indices_to_merge, 
                                      event_params['parent position'])

    def ancestral_path(self, i):
        if self.d == 2:
            X = self.coalescent.labels[:,i,0]
            Y = self.coalescent.labels[:,i,1]
            return X, Y
        elif self.d == 1:
            return self.coalescent.labels[:,i,0]
        
    def display_trajectory(self):
        plt.figure()
        ax = plt.axes()
        if self.d == 2:
            for i in range(self.n):
                X, Y = self.ancestral_path(i)
                ax.plot(X, Y)
        elif self.d == 1:
            for i in range(self.n):
                X = self.ancestral_path(i)
                ax.plot(self.times, X)
        else:
            print("Not implemented.")

class SLFV_ARG(SLFV_dual):
    def __init__(self, event_dist, dim, genome_length, record_locii = None):
        super().__init__(event_dist, dim)
        assert genome_length > 0
        self.G = genome_length
        if record_locii is not None:
            self.record = True
            self.locii = np.array(record_locii)
        else:
            self.record = False
    
    def run_coalescent(self, lineages_init_positions, T,
                       record_IBD_segments = False,
                       min_segment_length = None,
                       verbose = False):
        if record_IBD_segments:
            assert min_segment_length > 0
        self.record_IBD_segments = record_IBD_segments
        self.min_segment_length = min_segment_length
        super().run_coalescent(lineages_init_positions, T, verbose=verbose)
        if record_IBD_segments:
            self.ARG.IBD_segments['length'] = self.ARG.IBD_segments['endpoint'] - \
                self.ARG.IBD_segments['start']
        
    def init_coalescent(self, lineages_init_positions):
        if self.record:
            self.ARG = ARG.AncestralRecombinationGraph(self.G, self.n, self.locii,
                                                      labels = lineages_init_positions)
        else:
            self.ARG = ARG.GenomePartition(self.G, self.n, labels = lineages_init_positions)
    
    def get_current_positions(self):
        return self.ARG.labels.values
    
    def merge(self, time, indices_to_merge, event_params, verbose = False):
        parent_positions = np.vstack((event_params['1st parent position'],
                                      event_params['2nd parent position']))
        lineage_indices = self.ARG.labels.index[indices_to_merge]
        self.ARG.merge_lineages(lineage_indices,
                                newlabels= parent_positions,
                                record_IBD_segments=self.record_IBD_segments,
                                min_segment_length=self.min_segment_length,
                                verbose = verbose)
        if self.record_IBD_segments:
            self.ARG.drop_lineages(self.min_segment_length)
    
    def get_IBD_segments(self):
        return self.ARG.IBD_segments
    
            
if __name__ == "__main__":
    ##dual = SLFV_dual_heterogeneous_dispersal(L = 200, r_left = .8, r_right = 2, u = .3)
    dim = 1
    radii = [1, 3]
    impacts = [1, 1]
    
    T = 1000
    n = 10
    lineages_init_positions = np.reshape(np.random.normal(0, 5, size = dim * n), (n, dim))
    event_dist = events.DualHeterogeneous(impacts, radii, dim)
    dual = SLFV_dual(event_dist, dim)
    dual.run_coalescent(lineages_init_positions, T)
    dual.display_trajectory()
    plt.grid(True)