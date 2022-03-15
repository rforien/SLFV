#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:56:07 2022

@author: raphael
"""

import numpy as np
from abc import ABC, abstractmethod

class EventDist(ABC):
    def __init__(self):
        self.rate = 1
        
    def intensity(self, shape):
        return self.rate * np.prod(shape)
        
    @abstractmethod
    def run_event(self, frequency):
        return frequency

class FixedEvent(EventDist):
    def __init__(self, impact_parameter, radius):
        super().__init__()
        assert impact_parameter > 0 and impact_parameter <= 1
        assert radius > 0
        self.u = impact_parameter
        self.r = radius
    
    def _pick_k(self, freq):
        return np.random.choice(np.arange(np.size(freq)), p = freq)
    
    def run_event(self, frequency, impact = None, radius = None):
        if impact is None:
            impact = self.u
        if radius is None:
            radius = self.r
        centre = np.random.rand(np.size(frequency.shape))*frequency.shape
        freq = frequency.freq_in_ball(centre, radius)
        k = self._pick_k(freq)
        frequency.update(centre, radius, impact, k)

class FixedEventWithMutation(FixedEvent):
    def __init__(self, impact_parameter, radius, mutation_proba):
        super().__init__(impact_parameter, radius)
        assert mutation_proba >= 0 and mutation_proba <= 1
        self.mu = mutation_proba
    
    def _pick_k(self, freq):
        if np.random.rand() <= self.mu:
            freq = np.ones(np.size(freq))/np.size(freq)
        return super()._pick_k(freq)

class StableRadius(FixedEvent):
    def __init__(self, impact_parameter, min_radius, alpha, dimension):
        super().__init__(impact_parameter, min_radius)
        assert alpha > 0
        assert type(dimension)==int and dimension >= 1
        self.a = alpha + dimension
        self.rate = 1/(min_radius)**(self.a)
    
    def run_event(self, frequency):
        radius = self.r * np.random.rand()**(-1/self.a)
        super().run_event(frequency, radius = radius)

class OneTailRadii(FixedEvent):
    def __init__(self, u0, r0, c, b, a, dim):
        FixedEvent.__init__(self, u0, r0)
        assert c >= 0 and b >= 0 and a > 0
        assert type(dim) == int and dim >= 1
        self.tail = dim + a - c
        self.c = c
        self.b = b
        self.rate = 1/(self.r)**(-1/self.tail)
    
    def run_event(self, frequency):
        r2 = self.r * np.random.rand()**(-1/self.tail)
        r1 = r2**self.b
        u = self.u * r2**(-self.c)
        centre = np.random.rand(np.size(frequency.shape))*frequency.shape
        freq = frequency.freq_in_ball(centre, r1)
        try:
            k = self._pick_k(freq)
        except ValueError:
            i = 0
            while np.mean(freq)==np.nan:
                i=i+1
                freq = frequency.freq_in_ball(centre, r1 * (1+i/2))
            k = self._pick_k(freq)
        frequency.update(centre, r2, u, k)
        
class OneTailRadiiWithMutation(OneTailRadii,FixedEventWithMutation):
    def __init__(self, u0, r0, c, b, a, dim, mutation_proba):
        OneTailRadii.__init__(self, u0, r0, c, b, a, dim)
        FixedEventWithMutation.__init__(self, u0, r0, mutation_proba)
        
