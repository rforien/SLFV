#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:56:07 2022

@author: raphael
"""

import numpy as np
from abc import ABC, abstractmethod
import scipy.special as sp

class EventDist(ABC):
    def __init__(self):
        self.rate = 1
        
    def intensity(self, shape):
        return self.rate * np.prod(shape)
        
    @abstractmethod
    def run_event(self, frequency):
        return frequency

class DualEventDist(ABC):
    def __init__(self, dim, params):
        assert type(dim) is int and dim >= 1
        self.rate = 1
        self.d = dim
    
    def jump_rates(self, lineages_positions):
        return self.rate * np.ones(np.size(lineages_positions, axis = 0))
    
    @abstractmethod
    def draw_event_params(self, lineage_position):
        '''
        Returns the centre, parent position, impact and radius of an 
        event, given that it involves a lineage at the specified 
        position (before thinning).

        Parameters
        ----------
        lineage_position : numpy array
            position of the lineage involved in the event.

        Returns
        -------
        params: dict
            parameters needed to run an event.

        '''
        return {}


class FixedEvent(EventDist):
    def __init__(self, impact_parameter, radius):
        super().__init__()
        assert impact_parameter > 0 and impact_parameter <= 1
        assert radius > 0
        self.u = impact_parameter
        self.r = radius
    
    def _pick_k(self, freq):
        return np.random.choice(np.arange(np.size(freq)), p = freq)
    
    def run_event(self, frequency, impact = None, radius = None, centre = None):
        if impact is None:
            impact = self.u
        if radius is None:
            radius = self.r
        if centre is None:
            centre = np.random.rand(np.size(frequency.shape))*frequency.shape
        freq = frequency.freq_in_ball(centre, radius)
        k = self._pick_k(freq)
        frequency.update(centre, radius, impact, k)

def uniform_draw_from_ball(centre, radius):
    dim = np.size(centre)
    X = np.random.normal(0, 1, size = dim)
    direction = X / np.sqrt(np.sum(X**2))
    r = np.random.uniform()**(1/dim) * radius
    return centre + r * direction

def ball_volume(radius, dim):
    return np.pi ** (dim / 2) * radius ** dim / sp.gamma(1 + dim / 2)

class IgnoreEvent(Exception):
    pass

class DualFixedEvent(DualEventDist):
    def __init__(self, dim, params):
        super().__init__(dim, params)
        assert params['impact'] > 0 and params['impact'] <= 1
        assert params['radius'] > 0
        self.u = params['impact']
        self.r = params['radius']
    
    def draw_event_params(self, lineage_position):
        centre = uniform_draw_from_ball(lineage_position, self.r)
        parent_position = uniform_draw_from_ball(centre, self.r)
        params = {'centre': centre,
                  'radius': self.r,
                  'impact': self.u,
                  'parent position' : parent_position}
        return params

class TwoParentFixedRadius(DualFixedEvent):
    def draw_event_params(self, lineage_position):
        params = super().draw_event_params(lineage_position)
        params['1st parent position'] = params['parent position']
        second_parent = uniform_draw_from_ball(params['centre'],
                                               params['radius'])
        params['2nd parent position'] = second_parent
        return params

class DualHeterogeneous(DualEventDist):
    def __init__(self, dim, params):
        super().__init__(dim, params)
        assert np.min(params['impacts']) > 0 and np.max(params['impacts']) <= 1
        assert np.min(params['radii']) > 0
        self.impacts = np.array(params['impacts'])
        self.radii = np.array(params['radii'])
        self.R = np.max(self.radii)
        self.signs = np.array([-1, 1])
    
    def jump_rates(self, lineages_positions):
        if self.d > 1:
            first_coords = lineages_positions[:,0]
        else:
            first_coords = lineages_positions
        rates = np.ones(np.size(first_coords)) + (np.abs(first_coords) < self.R).astype(float)
        return rates
    
    def draw_event_params(self, lineage_position):
        if self.d > 1:
            x = lineage_position[0]
        else:
            x = lineage_position
        p = 0.5 * (float(x > - self.R) + float(x > self.R))
        a = np.random.binomial(1, p)
        centre = uniform_draw_from_ball(lineage_position, self.radii[a])
        if self.signs[a] * centre[0] < 0:
            raise IgnoreEvent()
        parent_position = uniform_draw_from_ball(centre, self.radii[a])
        params = {'centre': centre,
                  'radius': self.radii[a],
                  'impact': self.impacts[a],
                  'parent position': parent_position}
        return params

class TwoParentHeterogeneous(DualHeterogeneous, TwoParentFixedRadius):
    def __init__(self, dim, params):
        # add impact and radius values so that FixedRadius constructor runs smoothly
        params['radius'] = 1
        params['impact'] = 0.5
        super().__init__(dim, params)
    
    def draw_event_params(self, lineage_position):
        params = DualHeterogeneous.draw_event_params(self, lineage_position)
        params['1st parent position'] = params['parent position']
        second_parent = uniform_draw_from_ball(params['centre'],
                                               params['radius'])
        params['2nd parent position'] = second_parent
        return params
        

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
        r1 = self.r * (r2/self.r)**self.b
        u = self.u * (r2/self.r)**(-self.c)
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
        
