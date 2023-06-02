#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:14:29 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class SegmentList(object):
    def __init__(self, starts, ends, individuals, lineages):
        self.starts = starts
        self.ends = ends
        self.individuals = individuals
        self.lineages = lineages
    
    def __len__(self):
        return np.size(self.starts)
    
    def from_DataFrame(self, df):
        self.starts = df['start']
        self.ends = df['end']
        self.individuals = df['individual']
        self.lineages = df['lineage']
    
    def to_DataFrame(self):
        df = pd.DataFrame(data = {'individual': self.individuals,
                                        'lineage': self.lineages,
                                        'start': self.starts,
                                        'end': self.ends})
        return df
    
    def iterate(self):
        return zip(self.individuals, self.lineages, self.starts, self.ends)
    
    def fold_back(self, G):
        self.starts = np.mod(self.starts, G)
        self.ends = G - np.mod(-self.ends, G)
    
    def join(self, index, new_segments):
        self.starts = np.hstack((self.starts[index], new_segments.starts))
        self.ends = np.hstack((self.ends[index], new_segments.ends))
        self.individuals = np.hstack((self.individuals[index], new_segments.individuals))
        self.lineages = np.hstack((self.lineages[index], new_segments.lineages))
    
    def drop(self, index):
        self.starts = self.starts[~index]
        self.ends = self.ends[~index]
        self.individuals = self.individuals[~index]
        self.lineages = self.lineages[~index]

class RecombinationPattern(object):
    def __init__(self, startpoints, endpoints, lineages):
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.lineages = lineages
    
    def __len__(self):
        return np.size(self.startpoints)

class IBDSegmentList(object):
    def __init__(self, pairs, starts, ends):
        self.pairs = pairs
        self.starts = starts
        self.ends = ends
    
    def __len__(self):
        return np.size(self.starts)
    
    def add(self, pairs, starts, ends):
        self.pairs = np.concatenate((self.pairs, pairs), axis = 0)
        self.starts = np.concatenate((self.starts, starts))
        self.ends = np.concatenate((self.ends, ends))
    
    def to_DataFrame(self):
        df = pd.DataFrame(data = {'individual1': self.pairs[:,0],
                                  'individual2': self.pairs[:,1],
                                  'start': self.starts,
                                  'end': self.ends,
                                  'length': self.ends - self.starts})
        return df

class GenomePartition(object):
    def __init__(self, genome_length, n, segments = None, labels = None):
        assert type(n) == int and n >= 1
        self.n = int(n)
        assert genome_length >= 0
        self.G = genome_length
        self.init_segments(segments)
        self.init_labels(labels)
        self.steps = ['drawing recombination pattern',
                      'splitting segments',
                      'finding ibd segments',
                      'updating segment dataframe',
                      'updating labels']
        self.totals = np.zeros(len(self.steps))
        self.ntics = 0
    
    def __len__(self):
        return len(self.lineages)
    
    def init_segments(self, segments = None):
        if segments is not None:
            assert type(segments) is pd.DataFrame
            self.segments = SegmentList.from_DataFrame(segments)
            self.lineages = np.unique(self.segments.lineages)
        else:
            self.segments = SegmentList(starts = np.zeros(self.n), 
                                        ends = self.G * np.ones(self.n), 
                                        individuals = np.arange(self.n), 
                                        lineages = np.arange(self.n))
            self.lineages = np.arange(self.n)
    
    def init_labels(self, labels = None):
        if labels is not None:
            self.labels = np.array(labels)
            assert np.size(self.labels, axis = 0) == len(self), "Wrong number of labels"
        else:
            self.labels = None
    
    def get_segments(self):
        return self.segments.to_DataFrame()
    
    def partition(self, locus):
        at_locus = np.logical_and(self.segments.starts <= locus,
                                  self.segments.ends > locus)
        inverse = np.argsort(self.segments.individuals[at_locus])
        partition = self.segments.lineages[at_locus][inverse]
        return partition
    
    def labels_at_locus(self, locus):
        partition = self.partition(locus)
        indexes = np.array([np.where(self.lineages == l)[0][0] for l in partition])
        return self.labels[indexes]
    
    def plot_partition(self, ax = None):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for (individual, lineage, start, end) in self.segments.iterate():
            y = individual
            y = [y, y]
            x = [start, end]
            color = colors[np.mod(lineage, len(colors)).astype(int)]
            ax.plot(x, y, color = color, linewidth = 2)
        return ax
    
    def plot_segments(self, ax = None, linewidth = 2, fontsize = 12, jitter = 0.1):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in self.lineages:
            ax.plot([0, self.G], [i, i], linewidth = linewidth * 0.8, color = 'black')
        for (individual, lineage, start, end) in self.segments.iterate():
            y = lineage + np.random.normal(0, jitter)
            y = [y, y]
            x = [start, end]
            color = colors[np.mod(individual, len(colors)).astype(int)]
            ax.plot(x, y, linewidth = linewidth, color = color)
        return ax
        
    def draw_recombination_pattern(self, nb_recombining, new_lineages):
        assert type(nb_recombining) is int and nb_recombining > 0
        nb_cross_overs = np.random.poisson(self.G * nb_recombining)
        cross_overs = np.random.uniform(0, self.G * nb_recombining, size = nb_cross_overs)
        parent_offset = np.random.binomial(1, 0.5, size = nb_recombining)
        
        startpoints = np.sort(np.hstack((cross_overs, self.G * np.arange(nb_recombining))))
        id_recombining = (startpoints // self.G).astype(int)
        parents = np.mod(parent_offset[id_recombining] + np.arange(nb_recombining + nb_cross_overs), 2).astype(int)
        lineages = np.array(new_lineages)[parents]
        endpoints = np.hstack((startpoints[1:], [self.G * nb_recombining]))
        R = RecombinationPattern(startpoints, endpoints, lineages)
        return R
    
    def split_segments(self, lineages, rec_pattern):
        rec_index = np.isin(self.segments.lineages, lineages)
        # shift the segments belonging to different individuals so that they each recombine independently
        offset = self.G * (np.cumsum(np.isin(np.arange(np.max(lineages)+1), lineages))-1)
        starts = self.segments.starts[rec_index] + offset[self.segments.lineages[rec_index]]
        ends = self.segments.ends[rec_index] + offset[self.segments.lineages[rec_index]]
        n_seg = np.sum(rec_index)
        n_rec = len(rec_pattern)
        individuals = np.tile(self.segments.individuals[rec_index], n_rec)
        # compute start and end for new segments
        starts = np.maximum(np.tile(starts, n_rec),
                            np.repeat(rec_pattern.startpoints, n_seg))
        ends = np.minimum(np.tile(ends, n_rec),
                          np.repeat(rec_pattern.endpoints, n_seg))
        lineages = np.repeat(rec_pattern.lineages, n_seg)
        positive_length = (starts < ends).astype(bool)
        starts = starts[positive_length]
        ends = ends[positive_length]
        lineages = lineages[positive_length]
        individuals = individuals[positive_length]
        return SegmentList(starts, ends, individuals, lineages)
    
    def get_new_lineages_indexes(self, remaining_index):
        indexes = np.arange(len(self) + 2)
        remaining = self.lineages[remaining_index]
        newlineages = indexes[~np.isin(indexes, remaining)][:2]
        return newlineages
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       save_patterns = False,
                       record_IBD_segments = False,
                       min_segment_length = None,
                       verbose = False):
        # tics = [time.time()]
        if len(lineages_to_merge) == 0:
            return
        remaining_index = ~np.isin(self.lineages, lineages_to_merge)
        new_lineages = self.get_new_lineages_indexes(remaining_index)
        pattern = self.draw_recombination_pattern(len(lineages_to_merge), new_lineages)
        if save_patterns:
            if not hasattr(self, 'patterns'):
                self.patterns = []
            self.patterns.append(pattern)
        # tics.append(time.time())
        new_segments = self.split_segments(lineages_to_merge, pattern)
        # tics.append(time.time())
        if record_IBD_segments:
            self.find_ibd(new_segments, min_segment_length, verbose = verbose)
        # tics.append(time.time())
        # undo the offset created during the recombination
        new_segments.fold_back(self.G)
        # update segments
        self.segments.join(~np.isin(self.segments.lineages, lineages_to_merge),
                           new_segments)
        # update lineages
        remaining_new = np.isin(new_lineages, new_segments.lineages)
        self.lineages = np.hstack((self.lineages[remaining_index], 
                                   new_lineages[remaining_new]))
        # tics.append(time.time())
        # update labels
        if self.labels is not None:
            newlabels = np.array(newlabels)
            self.labels = np.concatenate((self.labels[remaining_index,:], 
                                          newlabels[remaining_new]))
        # tics.append(time.time())
        # self.totals = self.totals + np.diff(tics)
        # self.ntics = self.ntics + 1
        # percents = 100 * self.totals / np.sum(self.totals)
        # tot = np.sum(self.totals) / self.ntics
        # print("Number of lineages: %d." % len(self.lineages))
        # for dt, step in zip(percents, self.steps):
        #     print(step + ': %f percent.' % dt)
        # print("Total: %f s." % tot)
    
    def find_ibd(self, segments, min_segment_length, verbose = False):
        if not hasattr(self, 'IBD_segments'):
            self.IBD_segments = IBDSegmentList(np.zeros((0,2)), np.array([]), np.array([]))
        if len(segments) == 0:
            return
        if np.max(segments.starts) < self.G:
            return
        assert min_segment_length > 0
        # remove offsets
        startpoints = np.mod(segments.starts, self.G)
        endpoints = self.G - np.mod(-segments.ends, self.G)
        # compute lengths of segments
        S1, S2 = np.meshgrid(startpoints, startpoints)
        E1, E2 = np.meshgrid(endpoints, endpoints)
        Start = np.maximum(S1, S2)
        End = np.minimum(E1, E2)
        lengths = End - Start
        lengths = np.maximum(lengths, 0)
        # keep segments who end up in the same lineage
        L1, L2 = np.meshgrid(segments.lineages, segments.lineages)
        lengths = lengths * (L1 == L2)
        # keep segments who come from different lineages just before the event
        previous_lineage = segments.starts // self.G
        pL1, pL2 = np.meshgrid(previous_lineage, previous_lineage)
        lengths = lengths * (pL1 != pL2)
        # remove duplicates
        index = np.arange(len(segments))
        i, j = np.meshgrid(index, index)
        lengths = lengths * (i < j)
        # retrieve indices of segments longer than threshold length
        pairs = np.argwhere(lengths > min_segment_length)
        starts = Start[pairs[:,0], pairs[:,1]]
        ends = End[pairs[:,0], pairs[:,1]]
        individuals = segments.individuals[pairs]
        self.IBD_segments.add(individuals, starts, ends)
        if verbose:
            print("Found %d ibd segments so far." % len(self.IBD_segments))
    
    def drop_lineages(self, min_segment_length):
        self.segments.drop(self.segments.ends - self.segments.starts < min_segment_length)
        remaining = np.isin(self.lineages, self.segments.lineages)
        self.lineages = self.lineages[remaining]
        if self.labels is not None:
            self.labels = self.labels[remaining]

class AncestralRecombinationGraph(GenomePartition):
    def __init__(self, genome_length, n, locii, segments = None, labels = None):
        locii = np.unique(locii)
        assert locii.ndim == 1 and np.max(locii) < genome_length
        super().__init__(genome_length, n, segments = segments, labels = labels)
        self.locii = locii
        self.partition_record = np.array([[self.partition(locus) for locus in self.locii]])
        if self.labels is not None:
            self.label_record = np.array([[self.labels_at_locus(l) for l in self.locii]])
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       save_patterns = False,
                       record_IBD_segments = False,
                       min_segment_length = None,
                       verbose = False):
        super().merge_lineages(lineages_to_merge, newlabels, save_patterns,
                               record_IBD_segments, min_segment_length, verbose)
        self.partition_record = np.vstack((self.partition_record,
                                           np.array([[self.partition(locus) for locus in self.locii]])))
        if self.labels is not None:
            self.label_record = np.vstack((self.label_record,
                                           np.array([[self.labels_at_locus(l) 
                                                     for l in self.locii]])))
    
    def drop_lineages(self, min_segment_length):
        print("Warning: cannot drop lineages in ARG.")
        pass

    def plot_traj(self, project_against = None, jitter = 0.05, times = None, 
                  locii = None):
        if locii is None:
            locii = self.locii
        else:
            locii = np.array(locii)[np.isin(locii, self.locii)]
            assert np.size(locii) > 0
        d = np.size(self.label_record, axis = -1)
        if project_against is None:
            project_against = np.hstack(([1], np.zeros(d-1)))
        else:
            assert np.size(project_against) == d
            project_against = np.array(project_against)
            project_against = project_against / np.sqrt(np.sum(project_against**2))
        nt = np.size(self.partition_record, axis = 0)
        if times is None:
            times = np.arange(nt)
        else:
            assert np.size(times) == nt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [colors[int(i)] for i in np.mod(np.arange(len(self.locii)), len(colors)).astype(int)]
        plt.figure()
        ax = plt.axes()
        lines = []
        for i in range(self.n):
            for lo in locii:
                l = np.where(self.locii == lo)[0][0]
                projection = np.sum(self.label_record[:,l,i,:] * project_against[np.newaxis, :],
                                    axis = 1)
                # we add a small noise to be able to distinguish lineages
                jit = np.random.normal(0, jitter, size = nt)
                y = projection + jit
                x = times
                # y = np.repeat(y, 2)[:-1]
                # x = np.repeat(times, 2)[1:]
                line, = ax.plot(x, y, color = colors[l])
                if i == 0:
                    lines.append(line)
        ax.legend(lines, 
                   ['locus %.2f' % l for l in locii],
                   loc = 'best')
        ax.set_xlabel('time')
        ax.set_ylabel('position of the lineages')
        return ax

'''
class _AncestralRecominationGraph(object):
    def __init__(self, genome_length, n, labels = None):
        self.GPs = [GenomePartition(genome_length, n, labels = labels)]
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       record_IBD_segments = False,
                       min_segment_length = None,):
        if record_IBD_segments:
            GP, IBD_segments = self.GPs[-1].merge_lineages(lineages_to_merge, 
                                                           newlabels,
                                                           return_IBD_segments = True,
                                                           min_segment_length=min_segment_length,
                                                           inplace = False)
        else:
            GP = self.GPs[-1].merge_lineages(lineages_to_merge, newlabels, inplace = False)
        self.GPs.append(GP)
        if record_IBD_segments:
            if not hasattr(self, 'IBD_segments'):
                self.IBD_segments = IBD_segments
            else:
                self.IBD_segments = pd.concat((self.IBD_segments,
                                               IBD_segments),
                                              ignore_index=True)
    

    
    def _get_labels(self):
        return self.GPs[-1].labels
    def _set_labels(self):
        pass
    labels = property(_get_labels, _set_labels)
    
    def plot_segments(self):
        self.GPs[-1].plot_segments()
        
    def plot_partition(self):
        self.GPs[-1].plot_partition()
    
    def drop_lineages(self, min_segment_length):
        self.GPs[-1] = self.GPs[-1].drop_lineages(min_segment_length)
'''

if __name__ == '__main__':
    n = 2
    positions = np.random.normal(0, 1, size = (n,2))
    G = 2
    locii = np.linspace(0, G, 5, endpoint=False)
    arg = AncestralRecombinationGraph(G, n, locii, labels = positions)
    
    n_events = 3
    for i in range(n_events):
        new_positions = arg.labels + np.random.normal(0, 1, size = (2, 2))
        arg.merge_lineages([0, 1], newlabels = new_positions)
    
    arg.plot_traj(project_against=np.array([1, 1]))
    
    
    # GP = GenomePartition(genome_length=4, n=2)
    # # GP.plot_segments()
    # GP.merge_lineages([0])
    # segments = GP.merge_lineages([0, 1], return_IBD_segments=True, min_segment_length=0.01)
    # # GP, _ = GP.merge_lineages([3, 1, 2], return_IBD_segments = True, min_segment_length=0.01)
    # GP.plot_segments()
    # print(segments)