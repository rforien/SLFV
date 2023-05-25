#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:14:29 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GenomePartition(object):
    def __init__(self, genome_length, n, segments = None, labels = None):
        assert type(n) == int and n >= 1
        if n > 50:
            raise Exception("Not implemented for samples larger than 50.")
        self.n = int(n)
        assert genome_length >= 0
        self.G = genome_length
        if segments is None:
            self.init_segments(self.n)
        else:
            assert isinstance(segments, pd.DataFrame)
            self.segments = segments
        if labels is not None:
            if not (isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series)):
                labels = pd.DataFrame(data = labels)
            lineages = self.lineages
            # drop labels of lineages who do not carry any segments
            self.labels = labels.loc[np.isin(labels.index, lineages)]
            assert np.isin(lineages, self.labels.index).all(), "Missing labels"
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.lineages)
    
    def _get_lineages(self):
        return np.unique(self.segments['lineage'].values)
    def _set_lineages(self, l):
        pass
    lineages = property(_get_lineages, _set_lineages)
    
    def init_segments(self, n):
        self.segments = pd.DataFrame(data = {'individual': np.arange(self.n),
                                             'lineage': np.arange(self.n),
                                             'start': np.zeros(self.n),
                                             'end': self.G * np.ones(self.n)})
    
    def partition(self, locus):
        segments_at_locus = self.segments.loc[np.logical_and(self.segments['start'] <= locus,
                                                             self.segments['end'] > locus)]
        partition = segments_at_locus.set_index('individual')['lineage']
        return partition.sort_index()
    
    def plot_partition(self, ax = None):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for (index, segment) in self.segments.iterrows():
            y = segment['individual']
            y = [y, y]
            x = [segment['start'], segment['end']]
            color = colors[np.mod(segment['lineage'], len(colors)).astype(int)]
            ax.plot(x, y, color = color, linewidth = 2)
        return ax
    
    def plot_segments(self, ax = None, linewidth = 2, fontsize = 12):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in self.lineages:
            ax.plot([0, self.G], [i, i], linewidth = linewidth * 0.8, color = 'black')
        for (index, segment) in self.segments.iterrows():
            y = segment['lineage'] + np.random.normal(0, 0.1)
            y = [y, y]
            x = [segment['start'], segment['end']]
            color = colors[np.mod(segment['individual'], len(colors)).astype(int)]
            ax.plot(x, y, linewidth = linewidth, color = color)
        return ax
        
    def draw_recombination_pattern(self, nb_recombining, new_lineages):
        assert type(nb_recombining) is int and nb_recombining > 0
        nb_cross_overs = np.random.poisson(self.G * nb_recombining)
        cross_overs = np.sort(np.random.uniform(0, self.G * nb_recombining, size = nb_cross_overs))
        parent_offset = np.random.binomial(1, 0.5, size = nb_recombining)
        
        startpoints = np.sort(np.hstack((cross_overs, self.G * np.arange(nb_recombining))))
        id_recombining = (startpoints // self.G).astype(int)
        parents = np.mod(parent_offset[id_recombining] + np.arange(nb_recombining + nb_cross_overs), 2).astype(int)
        lineages = np.array(new_lineages)[parents]
        endpoints = np.hstack((startpoints[1:], [self.G * nb_recombining]))
        R = pd.DataFrame(data = {'lineage': lineages,
                                 'start': startpoints,
                                 'end': endpoints})
        return R
    
    def split_segments(self, lineages, rec_pattern):
        segments = self.segments.loc[np.isin(self.segments['lineage'], lineages)]
        # shift the segments belonging to different individuals so that they each recombine independently
        offset = self.G * (np.cumsum(np.isin(np.arange(np.max(lineages)+1), lineages))-1)
        starts = segments['start'] + offset[segments['lineage']]
        ends = segments['end'] + offset[segments['lineage']]
        n_seg = len(segments)
        n_rec = len(rec_pattern)
        individuals = np.tile(segments['individual'], n_rec)
        # compute start and end for new segments
        starts = np.maximum(np.tile(starts, n_rec),
                            np.repeat(rec_pattern['start'], n_seg))
        ends = np.minimum(np.tile(ends, n_rec),
                          np.repeat(rec_pattern['end'], n_seg))
        lineages = np.repeat(rec_pattern['lineage'], n_seg)
        split_segments = pd.DataFrame(data = {'individual': individuals,
                                              'lineage': lineages,
                                              'start': starts,
                                              'end': ends})
        # keep only segments with positive length
        split_segments = split_segments.loc[split_segments['start'] < split_segments['end']]
        return split_segments
    
    def get_new_lineages_indexes(self, lineages_to_merge):
        indexes = np.arange(len(self) + 2)
        remaining = self.lineages[~np.isin(self.lineages, lineages_to_merge)]
        newlineages = indexes[~np.isin(indexes, remaining)][:2]
        return newlineages
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       save_patterns = False,
                       record_IBD_segments = False,
                       min_segment_length = None):
        new_lineages = self.get_new_lineages_indexes(lineages_to_merge)
        pattern = self.draw_recombination_pattern(len(lineages_to_merge), new_lineages)
        new_segments = self.split_segments(lineages_to_merge, pattern)
        if record_IBD_segments:
            self.find_ibd(new_segments, min_segment_length)
        # undo the offset created during the recombination
        new_segments.loc[:,'start'] = np.mod(new_segments['start'], self.G)
        new_segments.loc[:,'end'] = self.G - np.mod(-new_segments['end'], self.G)
        # update segments
        self.segments = pd.concat((self.segments.loc[~np.isin(self.segments['lineage'], lineages_to_merge)],
                                   new_segments), ignore_index = True)
        if self.labels is not None:
            if not (isinstance(newlabels, pd.Series) or isinstance(newlabels, pd.DataFrame)):
                newlabels = pd.DataFrame(newlabels)
            newlabels.index = new_lineages
            self.labels = pd.concat((self.labels.drop(lineages_to_merge),
                                     newlabels))
            self.labels.drop(self.labels.index[~np.isin(self.labels.index, self.lineages)],
                             inplace = True)
    
    def find_ibd(self, segments, min_segment_length):
        if not hasattr(self, 'IBD_segments'):
            self.IBD_segments = pd.DataFrame(columns = ['individual1', 'individual2', 'start', 'endpoint'])
        if segments.empty or np.max(segments['start']) < self.G:
            return
        assert min_segment_length > 0
        # remove offsets
        startpoints = np.mod(segments['start'], self.G)
        endpoints = self.G - np.mod(-segments['end'], self.G)
        # compute lengths of segments
        S1, S2 = np.meshgrid(startpoints, startpoints)
        E1, E2 = np.meshgrid(endpoints, endpoints)
        Start = np.maximum(S1, S2)
        End = np.minimum(E1, E2)
        lengths = End - Start
        lengths = np.maximum(lengths, 0)
        # keep segments who end up in the same lineage
        L1, L2 = np.meshgrid(segments['lineage'], segments['lineage'])
        lengths = lengths * (L1 == L2)
        # keep segments who come from different lineages just before the event
        previous_lineage = segments['start'] // self.G
        pL1, pL2 = np.meshgrid(previous_lineage, previous_lineage)
        lengths = lengths * (pL1 != pL2)
        # remove duplicates
        index = np.arange(len(segments))
        i, j = np.meshgrid(index, index)
        lengths = lengths * (i < j)
        # retrieve indices of segments longer than threshold length
        pairs = np.argwhere(lengths > min_segment_length)
        ind1 = segments['individual'].values[pairs[:,0]]
        ind2 = segments['individual'].values[pairs[:,1]]
        starts = Start[pairs[:,0], pairs[:,1]]
        ends = End[pairs[:,0], pairs[:,1]]
        new_segments = pd.DataFrame(data = {'individual1': ind1,
                                            'individual2': ind2,
                                            'start': starts,
                                            'endpoint': ends})
        self.IBD_segments = pd.concat((self.IBD_segments, new_segments), ignore_index=True)
    
    def drop_lineages(self, min_segment_length, inplace = True):
        if inplace:
            self.segments.drop(self.segments.index[self.segments['end'] - 
                                                   self.segments['start'] < min_segment_length],
                              inplace = True)
            return self
        else:
            segments = self.segments.loc[self.segments['end'] - self.segments['start'] >= min_segment_length]
            return GenomePartition(self.G, self.n, segments, self.labels)

class AncestralRecombinationGraph(GenomePartition):
    def __init__(self, genome_length, n, locii, segments = None, labels = None):
        locii = np.unique(locii)
        assert locii.ndim == 1 and np.max(locii) < genome_length
        super().__init__(genome_length, n, segments = segments, labels = labels)
        self.locii = locii
        self.partition_record = np.array([[self.partition(locus).values for locus in self.locii]])
        if self.labels is not None:
            self.label_record = np.array([[self.labels.loc[self.partition(locus).values] 
                                 for locus in self.locii]])
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       save_patterns = False,
                       record_IBD_segments = False,
                       min_segment_length = None):
        super().merge_lineages(lineages_to_merge, newlabels, save_patterns,
                               record_IBD_segments, min_segment_length)
        self.partition_record = np.vstack((self.partition_record,
                                           np.array([[self.partition(locus).values for locus in self.locii]])))
        if self.labels is not None:
            self.label_record = np.vstack((self.label_record,
                                           np.array([[self.labels.loc[self.partition(locus).values]
                                                      for locus in self.locii]])))
    
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