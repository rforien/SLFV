#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:15:57 2018

@author: rforien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from . import events
from . import ARG

class Labelled_Coalescent(object):
    '''
    Class to record the coalescent tree of a sample of individuals, in which
    lineages carry some labels. This is for a haploid model without recombination,
    so each individual has a single parent in the previous step.
    '''
    def __init__(self, labels):
        '''
        Constructor

        Parameters
        ----------
        labels : numpy.ndarray
            Array giving the labels of the sampled individuals. The length along
            the first axis of this array determines the number of individuals in 
            the sample.

        '''
        assert type(labels) == np.ndarray, "Unexpected type of labels."
        n = np.size(labels, axis = 0)
        self.coalescent = np.reshape(np.arange(n), (1,n))
        self.labels = np.reshape(labels, (1,) + np.shape(labels))
        self.times = [0]
    
    def single_merger(self, time, indices_to_merge, new_label):
        '''
        Add a new merger in the coalescent. The merger must include at least one
        individual. If only one individual is involved, then this can be used to
        update the label of an individual.

        Parameters
        ----------
        time : float
            Time at which the merger takes place.
        indices_to_merge : list
            List of indices of the lineages involved in the merge. After the
            merge, all the corresponding lineages have the same ancestor.
        new_label : same type as the labels
            Label of the lineage replacing those involved in the merge.

        '''
        merge = [i in indices_to_merge for i in self.coalescent[-1,]]
        if any(merge):
            self.coalescent = np.vstack((self.coalescent, self.coalescent[-1,]))
            self.labels = np.vstack((self.labels, np.reshape(self.labels[-1,], (1,) + np.shape(self.labels[-1,]))))
            self.labels[-1,merge,] = new_label
            self.coalescent[-1,merge] = np.min(indices_to_merge)
        self.times.append(time)
    
    def kill(self, index):
        '''
        Kill a lineage.

        Parameters
        ----------
        index : int
            Index of the lineage to kill.

        '''
        print("Warning, killing not maintained!")
        self.single_merger([index], np.reshape(np.nan*np.ones(np.size(self.labels[-1,0])), (1, np.size(self.labels[-1,0]))))
    
    def alive_lineages(self):
        '''
        Return the list of alive lineages.
        '''
        return [x for x in self.coalescent[-1,:] if self.current_labels()[x,0] < np.inf]
    
    def nb_alive_lineages(self):
        '''
        Return the number of alive lineages.
        '''
        return np.size(self.alive_lineages())
    
    def current_labels(self):
        '''
        Return the current labels of the ancestors of the sampled individuals.
        '''
        return self.labels[-1,]
    
    def current_coalescent(self):
        '''
        Return the indices of the ancestors of the sampled individuals.
        '''
        return self.coalescent[-1,]
    
    def nb_current_lineages(self):
        '''
        Return the current list of lineages (whether alive or not).
        '''
        return np.size(np.unique(self.coalescent[-1,]))
    
def dist(positions, centre):
    '''
    Compute the distances of a set of positions to some reference point.

    Parameters
    ----------
    positions : numpy.ndarray
        Array giving the positions. Should be of shape (n, d), where d is the
        dimension of the space.
    centre : numpy.ndarray
        Coordinates of the reference point. Should be of shape (d,).

    Returns
    -------
    distances : numpy.ndarray
        Array of shape (n,) giving the (euclidian) distances of the positions
        to the reference point.

    '''
    return np.sqrt(np.sum((positions - centre[np.newaxis,:])**2, axis = 1))

class SLFV_dual(object):
    '''
    Class to simulate the dual of the SLFV with a single parent at each
    reproduction event.
    
    The history of the process is recorded in a Labelled_Coalescent object,
    where the labels are the positions of the lineages.
    '''
    def __init__(self, event_dist, dim):
        '''
        Constructor

        Parameters
        ----------
        event_dist : instance of a class which inherits from DualEventDist
            Object used to draw the reproduction events.
        dim : int
            Dimension of geographical space in which the simulation takes place.

        '''
        assert type(dim) == int and dim >= 1
        self.d = dim
        self.event_dist = event_dist
    
    def run_coalescent(self, lineages_init_positions, T, verbose = False):
        '''
        Run the simulation for a specified amount of time.

        Parameters
        ----------
        lineages_init_positions : numpy.ndarray
            Coordinates of the initial positions of the lineages, of shape (n, d),
            where n is the number of lineages and d is the dimension of the space.
        T : float
            Time for which to run the simulation.
        verbose : Bool, optional
            If True, progress information will be printed. The default is False.

        '''
        assert np.size(lineages_init_positions, axis = 1) == self.d
        self.n = np.size(lineages_init_positions, axis = 0)
        self.init_coalescent(lineages_init_positions)
        self.times = [0]
        t = 0
        last_tic = time.time()
        # tics = [time.time()]
        # steps = ['ignoring events',
        #           'draw next event time',
        #           'draw event parameters',
        #           'draw involved lineages',
        #           'compute merge']
        # totals = np.zeros(len(steps))
        # n = 0
        while True:
            if verbose:
                print("Progress: %.1f%%" % (100*(t / T)))
            else:
                tic = time.time()
                if tic > last_tic + 3600:
                    print("Progress: %.1f%%" % (100*(t / T)))
                    last_tic = tic
            # tics.append(time.time())
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
            # tics.append(time.time())
            # draw the lineage involved in the event
            i = np.random.choice(np.arange(len(rates)), p = rates / np.sum(rates))
            try:
                params = self.event_dist.draw_event_params(positions[i])
            except events.IgnoreEvent:
                # tics = [tics[0]]
                continue
            # tics.append(time.time())
            lineages = np.arange(np.size(positions, axis = 0))
            in_ball = dist(positions, params['centre']) <= params['radius']
            nb_in_ball = np.sum(in_ball)
            if nb_in_ball == 1:
                indices_to_merge = np.array([i])
            else:
                in_ball[i] = False
                involved = np.random.binomial(1, params['impact'], 
                                              size = nb_in_ball-1).astype(bool)
                indices_to_merge = np.hstack(([i], lineages[in_ball][involved]))
                # thinning to avoid multiple counting
                if np.size(indices_to_merge) > 1:
                    if np.random.uniform() > 1/np.size(indices_to_merge):
                        if verbose:
                            print("Ignoring event")
                        # tics = [tics[0]]
                        continue
            # tics.append(time.time())
            # add merger
            self.merge(t, indices_to_merge, params, verbose = verbose)
            self.times.append(t)
            # tics.append(time.time())
            # totals = totals + np.diff(tics)
            # percents = totals / np.sum(totals) * 100
            # n = n + 1
            # print("Number of lineages: %d." % len(self.ARG.lineages))
            # for (dt, step) in zip(percents, steps):
            #     print(step + ': %f percent' % dt)
            # print("Total: %f" % (np.sum(totals)/n))
            # tics = [time.time()]
        self.times = np.array(self.times)
    
    def init_coalescent(self, lineages_init_positions):
        '''
        Initialise the coalescent object (can be overloaded in classes inheriting
        from this class to record the history in another form).
        '''
        self.coalescent = Labelled_Coalescent(lineages_init_positions)
    
    def get_current_positions(self):
        '''
        Return the current positions of the lineages.
        '''
        return self.coalescent.current_labels()
    
    def merge(self, merge_time, indices_to_merge, event_params, verbose = False):
        '''
        Merge a given set of lineages.

        Parameters
        ----------
        merge_time : float
            Time at which the merge takes place.
        indices_to_merge : list
            Indices of the lineages involved in the merge event.
        event_params : dict
            Parameters of the reproduction event. By default only the 
            'parent position' value is used.
        verbose : Bool, optional
            Not used by default, introduced for compatibility. 
            The default is False.

        '''
        self.coalescent.single_merger(merge_time,
                                      indices_to_merge, 
                                      event_params['parent position'])

    def ancestral_path(self, i):
        '''
        Return the trajectory of the lineage of a specified individual.

        Parameters
        ----------
        i : int
            index of the individual in the original sample.

        Returns
        -------
        X (, Y) : numpy.ndarray (, numpy.ndarray)
            Time series of the coordinates of the lineages along the trajectory.

        '''
        if self.d == 2:
            X = self.coalescent.labels[:,i,0]
            Y = self.coalescent.labels[:,i,1]
            return X, Y
        elif self.d == 1:
            return self.coalescent.labels[:,i,0]
        
    def display_trajectory(self):
        '''
        Plot the trajectory of the lineages.
        
        In two dimensions, the trace of each lineage is plotted in a different
        colour, with no time axis.
        In one dimension, the trajectories are plotted with time in the y axis
        and space in the x axis.

        '''
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
    '''
    Class to simulate the dual of the SLFV along a full genome using the ARG.
    '''
    def __init__(self, event_dist, dim, genome_length, record_locii = None):
        '''
        Constructor

        Parameters
        ----------
        event_dist : instance of a class inheriting from DualEventDist
            Event drawer used to draw the reproduction events.
        dim : int
            Dimension of the geographical space.
        genome_length : float
            Length, in Morgan, of the genome (only one chromosome is simulated).
        record_locii : list, optional
            If given, the process records the full history of the coalescent at
            the given list of locii. Otherwise only the current state is recorded. 
            The default is None.

        '''
        super().__init__(event_dist, dim)
        assert genome_length > 0
        self.G = genome_length
        if record_locii is not None:
            self.record = True
            self.locii = np.array(record_locii)
            assert np.min(self.locii) >= 0 and np.max(self.locii) < self.G
        else:
            self.record = False
    
    def run_coalescent(self, lineages_init_positions, T,
                       record_IBD_segments = False,
                       min_segment_length = None,
                       verbose = False,
                       dump_interval = None,
                       dump_filename = None):
        '''
        Run the coalescent for a specified amount of time.

        Parameters
        ----------
        lineages_init_positions : numpy.ndarray
            Coordinates of the initial positions of the lineages. Of shape (n, d),
            where n is the number of lineages, and d is the dimension of space.
        T : float
            Time for which to run the simulation.
        record_IBD_segments : Bool, optional
            If True, the IBD segments longer than min_segment_length will be 
            recorded. The default is False.
        min_segment_length : float, optional
            Minimum length (in Morgan) of the IBD segements to record.
            The default is None.
        verbose : Bool, optional
            If True, progress information will be displayed. The default is False.
        dump_interval : float, optional
            If passed, the current IBD segments will be saved every n seconds,
            under the named passed as dump_filename, where n is the value of
            dump_interval.
            The default is None.
        dump_filename : string, optional
            Name of the file to save the dumped IBD segments. The default is None.

        '''
        if record_IBD_segments:
            assert min_segment_length > 0
        if dump_interval is None:
            self.dump_interval = np.inf
        else:
            assert dump_interval > 0
            assert dump_filename is not None
            self.dump_interval = dump_interval
            self.dump_file_name = dump_filename
        self.record_IBD_segments = record_IBD_segments
        self.min_segment_length = min_segment_length
        self.dump_tic = time.time()
        super().run_coalescent(lineages_init_positions, T, verbose=verbose)
        if record_IBD_segments:
            self.IBD_segments = self.ARG.IBD_segments.to_DataFrame()
        
    def init_coalescent(self, lineages_init_positions):
        '''
        Initialise the coalescent object.
        '''
        if self.record:
            self.ARG = ARG.AncestralRecombinationGraph(self.G, self.n, self.locii,
                                                      labels = lineages_init_positions)
        else:
            self.ARG = ARG.GenomePartition(self.G, self.n, labels = lineages_init_positions)
    
    def get_current_positions(self):
        '''
        Return the current lineage positions
        '''
        return self.ARG.labels
    
    def merge(self, merge_time, indices_to_merge, event_params, verbose = False):
        '''
        Merge a given set of lineages.

        Parameters
        ----------
        merge_time : float
            Time at which the merge takes place.
        indices_to_merge : list
            Indices of the lineages involved in the merge event.
        event_params : dict
            Parameters of the reproduction event. The two values used are
            '1st parent position' and '2nd parent position'.
        verbose : Bool, optional
            Not used by default, introduced for compatibility. 
            The default is False.

        '''
        parent_positions = np.vstack((event_params['1st parent position'],
                                      event_params['2nd parent position']))
        lineages_to_merge = self.ARG.lineages[indices_to_merge]
        self.ARG.merge_lineages(lineages_to_merge,
                                newlabels= parent_positions,
                                record_IBD_segments=self.record_IBD_segments,
                                min_segment_length=self.min_segment_length,
                                verbose = verbose)
        if self.record_IBD_segments:
            self.ARG.drop_lineages(self.min_segment_length)
            dump_tic = time.time()
            if dump_tic > self.dump_tic + self.dump_interval:
                print("Dumping in %s." % self.dump_file_name)
                self.ARG.IBD_segments.to_DataFrame().to_csv(self.dump_file_name)
                self.dump_tic = dump_tic
    
    def get_IBD_segments(self):
        '''
        Return the current IBD segments
        '''
        return self.ARG.IBD_segments.to_DataFrame()
            
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