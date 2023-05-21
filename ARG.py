#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:14:29 2023

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from . import function as fct

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
        if blocks is not None:
            keep = np.array([False if block.maximum() == 0 else True
                             for block in blocks])
            self.blocks = np.array(blocks)[keep]
        else:
            self.init_blocks(self.n)
            keep = np.array([True for i in range(self.n)])
        if labels is not None:
            labels = labels[keep]
            assert np.size(labels, axis = 0) == len(self.blocks)
            self.labels = np.array(labels)
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.blocks)
    
    def init_blocks(self, n):
        x = np.array([0, self.G])
        self.blocks = []
        for i in range(self.n):
            seq = np.array(np.arange(self.n) == i)
            y = [to_number(seq), 0]
            self.blocks.append(fct.PiecewiseConstant(x, y))
        self.blocks = np.array(self.blocks)
    
    def partition(self, locus):
        partition = - np.ones(self.n).astype(int)
        for i in range(len(self)):
            n = self.blocks[i](locus)
            assert n < 2**self.n
            seq = to_sequence(n)
            children = np.where(seq == 1)[0]
            partition[children] = i
        return partition
    
    def check_sum(self):
        S = fct.PiecewiseConstant([0], [0])
        for b in self.blocks:
            S = S + b
        print(S.y / (2**self.n - 1))
    
    def plot_partition(self, ax = None):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        x = np.array([])
        for b in self.blocks:
            x = np.hstack((x, b.x))
        x = np.unique(x)
        for i in range(len(x)-1):
            partition = self.partition(x[i])
            for j in range(self.n):
                if partition[j] >= 0:
                    color = colors[np.mod(partition[j], len(colors))]
                    ax.plot(x[i:i+2], [j, j], color = color,
                            linewidth=2)
        return ax
    
    def plot_blocks(self, ax = None, linewidth = 2, fontsize = 12):
        if ax is None:
            plt.figure()
            ax = plt.axes()
        for i in range(len(self)):
            block = self.blocks[i]
            y = i
            for j in range(len(block.x)-1):
                if block.y[j]:
                    string = to_string(block.y[j])
                    line, = ax.plot(block.x[j:j+2], [y,y], linewidth = linewidth)
                    xmid = np.mean(block.x[j:j+2])
                    ax.annotate(string, (xmid, y), fontsize = fontsize,
                                color = line.get_color(),
                                xytext = (xmid, y + 0.05))
                else:
                    ax.plot(block.x[j:j+2], [y,y], linewidth = linewidth, 
                            color = 'black')
            if block.x[-1] < self.G:
                ax.plot([block.x[-1], self.G], [y,y], linewidth = linewidth, 
                        color = 'black')
        return ax
        
    def draw_recombination_pattern(self):
        nb_cross_overs = np.random.poisson(self.G)
        cross_overs = np.sort(np.random.uniform(0, self.G, size = nb_cross_overs))
        start = np.random.binomial(1, 0.5)
        x = np.hstack(([0], cross_overs))
        y = 0.5 * (1 - (-1)**(start + np.arange(nb_cross_overs + 1)))
        R = fct.PiecewiseConstant(x, y)
        return [R, -R + 1]
    
    def merge_lineages(self, lineages_to_merge, newlabels = None,
                       record_IBD_segments = False,
                       save_patterns = False,
                       return_IBD_segments = False):
        newblocks = np.array([fct.PiecewiseConstant([0], [0]),
                              fct.PiecewiseConstant([0], [0])])
        if save_patterns:
            self.patterns = []
        for l in lineages_to_merge:
            if l >= len(self):
                print("Warning, trying to merge non-existant lineage.")
                continue
            pattern = self.draw_recombination_pattern()
            if save_patterns:
                self.patterns.append(pattern[0])
            for i in range(2):
                newblocks[i] = self.merge_blocks(newblocks[i], 
                                                 pattern[i] * self.blocks[l],
                                                 return_IBD_segments)
        keep = ~np.isin(np.arange(len(self)), lineages_to_merge)
        blocks = np.hstack((self.blocks[keep], newblocks))
        if self.labels is not None:
            if len(np.shape(self.labels)) == 1:
                assert np.shape(newlabels) == (2,)
                labels = np.hstack((self.labels[keep], newlabels))
            else:
                assert np.size(newlabels, axis = 0) == 2
                labels = np.vstack((self.labels[keep], newlabels))
        else:
            labels = None
        new_GP = GenomePartition(self.G, self.n, blocks, labels)
        if return_IBD_segments:
            return new_GP, self.IBD_segments
        else:
            return new_GP
    
    def merge_blocks(self, block1, block2, return_IBD_segments = False):
        newblock = block1 + block2
        if not return_IBD_segments:
            return newblock
        IBD_segments = pd.DataFrame(columns = ['individual1', 'individual2', 'start', 'endpoint'])
        locus = 0
        while locus < self.G:
            next_points = np.hstack((block1.x[block1.x > locus],
                                     block2.x[block2.x > locus],
                                     [self.G]))
            next_locus = np.min(next_points)
            if block1(locus) * block2(locus) != 0:
                indices1 = to_indices(block1(locus))
                indices2 = to_indices(block2(locus))
                for i in indices1:
                    for j in indices2:
                        # check if the block is a continuation of a previous one
                        a = np.isin(IBD_segments['individual1'].values, [i,j])
                        b = np.isin(IBD_segments['individual2'].values, [i,j])
                        c = IBD_segments['endpoint'].values == locus
                        index = a * b * c
                        if index.any():
                            IBD_segments.loc[index, 'endpoint'] = next_locus
                        else:
                            new_segment = pd.Series({'individual1': i,
                                                     'individual2': j,
                                                     'start': locus,
                                                     'endpoint': next_locus})
                            IBD_segments = pd.concat((IBD_segments, new_segment.to_frame().T),
                                                     ignore_index=True)
            locus = next_locus
        if hasattr(self, 'IBD_segments'):
            self.IBD_segments = pd.concat((self.IBD_segments, IBD_segments),
                                          ignore_index=True)
        else:
            self.IBD_segments = IBD_segments
        return newblock
    
    def drop_lineages(self, min_segment_length):
        keep = []
        for i in range(len(self)):
            b = fct.PiecewiseConstant(self.blocks[i].x,
                                      self.blocks[i].y > 0)
            x = np.hstack((b.x, [self.G]))
            L = np.diff(x)[b.y > 0]
            if np.max(L) >= min_segment_length:
                keep.append(i)
        blocks = self.blocks[keep]
        if self.labels is not None:
            labels = self.labels[keep]
        else:
            labels = None
        return GenomePartition(self.G, self.n, blocks, labels)

class AncestralRecominationGraph(object):
    def __init__(self, genome_length, n, labels = None):
        self.GPs = [GenomePartition(genome_length, n, labels = labels)]
        self.times = [0]
    
    def add_merger(self, time, lineages_to_merge, newlabels = None,
                   record_IBD_segments = False):
        if record_IBD_segments:
            GP, IBD_segments = self.GPs[-1].merge_lineages(lineages_to_merge, 
                                                           newlabels,
                                                           return_IBD_segments = True)
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
    
    def drop_lineages(self, min_segment_length):
        self.GPs[-1] = self.GPs[-1].drop_lineages(min_segment_length)

if __name__ == '__main__':
    GP = GenomePartition(4, 4)
    GP = GP.merge_lineages([0])
    # GP = GP.merge_lineages([0, 1])
    # GP = GP.merge_lineages([3, 1, 2])
    GP.plot_blocks()