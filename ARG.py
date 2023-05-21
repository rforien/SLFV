#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:14:29 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def to_number(sequence):
    return np.sum(sequence * 2**np.arange(len(sequence))).astype(int)

def to_sequence(number):
    if number == 0:
        return np.array([])
    n = np.floor(np.log(number) / np.log(2)).astype(int)
    seq = np.zeros(n+1).astype(int)
    seq[n] = 1
    q = 2**n
    while q != number:
        # print(seq)
        i = np.floor(np.log(number - q) / np.log(2)).astype(int)
        seq[i] = 1
        q = q + 2**i
    # print(q, number)
    return seq

def to_indices(number):
    sequence = to_sequence(number)
    indices = np.arange(len(sequence))[sequence.astype(bool)]
    return indices

def to_string(number):
    string = ""
    indices = to_indices(number)
    for i in indices:
        string = string + str(i+1) + ", "
    string = r"$\lbrace %s \rbrace$" % string[:-2]
    return string

class GenomePartition(object):
    def __init__(self, genome_length, n, blocks = None, labels = None):
        assert type(n) == int and n >= 1
        if n > 50:
            raise Exception("Not implemented for samples larger than 50.")
        self.n = int(n)
        assert genome_length >= 0
        self.G = genome_length
        if blocks is None:
            self.init_blocks(self.n)
        else:
            assert isinstance(blocks, pd.DataFrame)
            self.blocks = blocks
        if labels is not None:
            if not (isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series)):
                labels = pd.DataFrame(data = labels)
            self.labels = labels.loc[np.isin(labels.index, self.lineages)]
            assert np.isin(self.lineages, self.labels.index).all(), "Missing labels"
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.lineages)
    
    def _get_lineages(self):
        return np.unique(self.blocks.index.get_level_values('lineage').values)
    def _set_lineages(self, l):
        pass
    lineages = property(_get_lineages, _set_lineages)
    
    def init_blocks(self, n):
        self.blocks = pd.DataFrame(index = pd.MultiIndex.from_tuples([(i, 0) for i in range(self.n)],
                                                                     names = ['lineage', 'ibd segment']),
                                   data = {'individual': np.arange(self.n),
                                           'start': np.zeros(self.n),
                                           'end': self.G * np.ones(self.n)})
    
    def partition(self, locus):
        blocks_at_locus = self.blocks.loc[np.logical_and(self.blocks['start'] <= locus,
                                                         self.blocks['end'] > locus)]
        index = np.argsort(blocks_at_locus['individual'].values)
        partition = blocks_at_locus.index.get_level_values('lineage')[index]
        return partition
    
    def plot_partition(self, ax = None):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for (index, block) in self.blocks.iterrows():
            lineage = index[0]
            y = block['individual']
            y = [y, y]
            x = [block['start'], block['end']]
            color = colors[np.mod(lineage, len(colors))]
            ax.plot(x, y, color = color, linewidth = 2)
        return ax
    
    def plot_blocks(self, ax = None, linewidth = 2, fontsize = 12):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in self.lineages:
            ax.plot([0, self.G], [i, i], linewidth = linewidth, color = 'black')
        for (index, block) in self.blocks.iterrows():
            y = index[0] + np.random.normal(0, 0.1)
            y = [y, y]
            x = [block['start'], block['end']]
            color = colors[np.mod(block['individual'], len(colors)).astype(int)]
            ax.plot(x, y, linewidth = linewidth, color = color)
        return ax
        
    def draw_recombination_pattern(self):
        nb_cross_overs = np.random.poisson(self.G)
        cross_overs = np.sort(np.random.uniform(0, self.G, size = nb_cross_overs))
        start = np.random.binomial(1, 0.5)
        parent = np.mod(start + np.arange(nb_cross_overs+1), 2).astype(int)
        startpoints = np.hstack(([0], cross_overs))
        endpoints = np.hstack((cross_overs, [self.G]))
        R = pd.DataFrame(data = {'parent': parent,
                                 'start': startpoints,
                                 'end': endpoints})
        return R
    
    def split_blocks(self, lineage, rec_pattern):
        blocks = self.blocks.loc[lineage, :]
        split_blocks = [pd.DataFrame(), pd.DataFrame()]
        for (i, segment) in rec_pattern.iterrows():
            start = np.maximum(blocks['start'], segment['start'])
            end = np.minimum(blocks['end'], segment['end'])
            new_blocks = pd.DataFrame(data = {'individual': blocks['individual'],
                                              'start': start,
                                              'end': end})
            new_blocks = new_blocks.loc[new_blocks['start'] < new_blocks['end']]
            parent = int(segment['parent'])
            split_blocks[parent] = pd.concat((split_blocks[parent],
                                              new_blocks),
                                             ignore_index=True)
        return split_blocks
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       save_patterns = False,
                       return_IBD_segments = False,
                       min_segment_length = None):
        newblocks = [pd.DataFrame(), pd.DataFrame()]
        indexes = np.arange(len(self) + 2)
        remaining = self.lineages[~np.isin(self.lineages, lineages_to_merge)]
        newlineages = indexes[~np.isin(indexes, remaining)][:2]
        if save_patterns:
            self.patterns = []
        for l in lineages_to_merge:
            if l not in self.lineages:
                print("Warning, trying to merge non-existant lineage.")
                continue
            pattern = self.draw_recombination_pattern()
            if save_patterns:
                self.patterns.append(pattern)
            split_blocks = self.split_blocks(l, pattern)
            for i in range(2):
                if return_IBD_segments:
                    self.find_ibd(newblocks[i], split_blocks[i], min_segment_length)
                newblocks[i] = pd.concat((newblocks[i], split_blocks[i]),
                                         ignore_index=True)
        for i in range(2):
            newblocks[i] = newblocks[i].assign(lineage=newlineages[i]).set_index('lineage', append = True).swaplevel(0,1)
        blocks = pd.concat((self.blocks.drop(index=lineages_to_merge, level='lineage'),
                            newblocks[0], newblocks[1]))
        if self.labels is not None:
            if not (isinstance(newlabels, pd.Series) or isinstance(newlabels, pd.DataFrame)):
                newlabels = pd.DataFrame(newlabels)
            newlabels.index = newlineages
            labels = pd.concat((self.labels.drop(lineages_to_merge),
                                     newlabels))
        else:
            labels = None
        new_GP = GenomePartition(self.G, self.n, blocks, labels)
        if return_IBD_segments:
            return new_GP, self.IBD_segments
        else:
            return new_GP
    
    def find_ibd(self, blocks1, blocks2, min_segment_length):
        if not hasattr(self, 'IBD_segments'):
            self.IBD_segments = pd.DataFrame(columns = ['individual1', 'individual2', 'start', 'endpoint'])
        if blocks1.empty or blocks2.empty:
            return
        assert min_segment_length > 0
        S1, S2 = np.meshgrid(blocks1['start'], blocks2['start'])
        E1, E2 = np.meshgrid(blocks1['end'], blocks2['end'])
        Start = np.maximum(S1, S2)
        End = np.minimum(E1, E2)
        lengths = End - Start
        lengths = np.maximum(lengths, 0)
        for pair in  np.argwhere(lengths > min_segment_length):
            segment = pd.Series({'individual1': blocks1['individual'][pair[1]],
                                 'individual2': blocks2['individual'][pair[0]],
                                 'start': Start[pair[0], pair[1]],
                                 'endpoint': End[pair[0], pair[1]]})
            self.IBD_segments = pd.concat((self.IBD_segments, segment.to_frame().T), ignore_index=True)
    
    def drop_lineages(self, min_segment_length):
        blocks = self.blocks.loc[self.blocks['end'] - self.blocks['start'] >= min_segment_length]
        return GenomePartition(self.G, self.n, blocks, self.labels)

class AncestralRecominationGraph(object):
    def __init__(self, genome_length, n, labels = None):
        self.GPs = [GenomePartition(genome_length, n, labels = labels)]
        self.times = [0]
    
    def add_merger(self, time, lineages_to_merge, newlabels = None,
                   record_IBD_segments = False,
                   min_segment_length = None):
        if record_IBD_segments:
            GP, IBD_segments = self.GPs[-1].merge_lineages(lineages_to_merge, 
                                                           newlabels,
                                                           return_IBD_segments = True,
                                                           min_segment_length=min_segment_length)
        else:
            GP = self.GPs[-1].merge_lineages(lineages_to_merge, newlabels)
        self.GPs.append(GP)
        self.times.append(time)
        if record_IBD_segments:
            if not hasattr(self, 'IBD_segments'):
                self.IBD_segments = IBD_segments
            else:
                self.IBD_segments = pd.concat((self.IBD_segments,
                                               IBD_segments),
                                              ignore_index=True)
    
    def get_current_labels(self):
        return self.GPs[-1].labels
    
    def plot_blocks(self):
        self.GPs[-1].plot_blocks()
        
    def plot_partition(self):
        self.GPs[-1].plot_partition()
    
    def drop_lineages(self, min_segment_length):
        self.GPs[-1] = self.GPs[-1].drop_lineages(min_segment_length)

if __name__ == '__main__':
    GP = GenomePartition(4, 4)
    # GP.plot_blocks()
    # GP = GP.merge_lineages([0])
    GP, _ = GP.merge_lineages([0, 1], return_IBD_segments=True, min_segment_length=0.01)
    # GP, _ = GP.merge_lineages([3, 1, 2], return_IBD_segments = True, min_segment_length=0.01)
    GP.plot_blocks()