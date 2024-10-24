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
    '''
    Abstract class to draw reproduction events in the (forwards-in-time) SLFV.
    
    The atribute rate should be defined in the constructor (the default behaviour
    is that this is the intensity of the poisson random measure driving the REs).
    
    The intensity method is used to compute the rate at which the next event takes
    place, and the method run_event (which needs to be overloaded) resolves a
    reproduction event.
    '''
    def __init__(self):
        self.rate = 1
        
    def intensity(self, shape):
        '''
        Compute the rate at which the next event takes place.

        Parameters
        ----------
        shape : tuple
            tuple giving the dimensions of the region on which the SLFV is 
            simulated.

        Returns
        -------
        rate : float
            rate at which the next event takes place.

        '''
        return self.rate * np.prod(shape)
        
    @abstractmethod
    def run_event(self, frequency):
        '''
        Run a reproduction event.

        Parameters
        ----------
        frequency : Frequency1D or Frequency2D
            Object giving the allelic frequencies of the model just before the
            reproduction event.

        Returns
        -------
        frequency : Frequency1D or Frequency2D
            Updated allelic frequencies just after the reproduciton event.

        '''
        return frequency

class IgnoreEvent(Exception):
    pass

class DualEventDist(ABC):
    '''
    Abstract class used to draw reproduction event in the SLFV dual.
    '''
    def __init__(self, dim, params):
        assert type(dim) is int and dim >= 1
        self.rate = 1
        self.d = dim
    
    def jump_rates(self, lineages_positions):
        '''
        Compute the jump rates of the lineages, given their positions. The rates
        can be overestimated at this step, an corrected by thinning later. In
        particular, the main SLFV_dual class performs an automatic thinning
        to account for events involving multiple lineages, so that the rates
        returned by this function should correspond to the total jumps rates of
        each lineage, ignoring the fact that several lineages may jump at the
        same time.

        Parameters
        ----------
        lineages_positions : numpy.ndarray
            Array of shape (n, d), where n is the current number of lineages and
            d is the dimension of the space, giving the locations of the lineages
            before the next event.

        Returns
        -------
        rates: numpy.ndarray
            Array of shape (n,) giving the jump rates of the lineages. All entries
            should be non-negative.

        '''
        return self.rate * np.ones(np.size(lineages_positions, axis = 0))
    
    @abstractmethod
    def draw_event_params(self, lineage_position):
        '''
        Returns the parameters of a reproduction event, given that it involves 
        a lineage at the specified position (before thinning).
        If the event should not be run (to perform some thinning), one should
        raise IgnoreEvent().

        Parameters
        ----------
        lineage_position : numpy.ndarray
            position of the lineage involved in the event (shape (d,)).

        Returns
        -------
        params: dict
            parameters needed to run an event.
            In the default SLFV_dual class, it should mention the centre, radius
            and parent location of the event.

        '''
        return {}


class FixedEvent(EventDist):
    '''
    Event drawer for the forwards-in-time SLFV with fixed impact and fixed radius.
    '''
    def __init__(self, impact_parameter, radius):
        '''
        Constructor

        Parameters
        ----------
        impact_parameter : float
            Impact of the events. Should be in (0, 1].
        radius : float
            Radius of the events. Should be positive.
        '''
        super().__init__()
        assert impact_parameter > 0 and impact_parameter <= 1
        assert radius > 0
        self.u = impact_parameter
        self.r = radius
    
    def _pick_k(self, freq):
        '''
        Pick a parental type according to some allelic frequencies.

        Parameters
        ----------
        freq : numpy.ndarray
            Allelic frequencies from which to pick.

        Returns
        -------
        k: int
            index of the parent allele.

        '''
        return np.random.choice(np.arange(np.size(freq)), p = freq)
    
    def run_event(self, frequency, impact = None, radius = None, centre = None):
        '''
        Run a reproduction event with fixed impact and radius.

        Parameters
        ----------
        frequency : Frequency1D or Frequency2D
            Object giving the allelic frequencies of the model just before the
            reproduction event.
        impact : float, optional
            Use this argument to override the default impact parameter.
        radius : flaot, optional
            Use this argument to override the default radius.
        centre : numpy.ndarray, optional
            Use this argument to specify the centre of the reproduction event.
            The default behaviour is to draw this uniformly from the whole area.

        Returns
        -------
        frequency: Frequency1D or Frequency2D
            Updated allelic frequencies just after the reproduciton event.

        '''
        if impact is None:
            impact = self.u
        if radius is None:
            radius = self.r
        if centre is None:
            centre = np.random.rand(np.size(frequency.shape))*frequency.shape
        freq = frequency.freq_in_ball(centre, radius)
        k = self._pick_k(freq)
        frequency.update(centre, radius, impact, k)
        return frequency

def uniform_draw_from_ball(centre, radius):
    '''
    Draw a point uniformly in a ball.

    Parameters
    ----------
    centre : numpy.ndarray
        Centre of the ball from which to draw.
    radius : float
        Radius of the ball (non-negative).

    Returns
    -------
    point: numpy.ndarray
        coordinates of the sampled point.

    '''
    dim = np.size(centre)
    X = np.random.normal(0, 1, size = dim)
    direction = X / np.sqrt(np.sum(X**2))
    r = np.random.uniform()**(1/dim) * radius
    return centre + r * direction

def ball_volume(radius, dim):
    '''
    Return the volume of the d-dimensional ball.
    '''
    return np.pi ** (dim / 2) * radius ** dim / sp.gamma(1 + dim / 2)


class DualFixedEvent(DualEventDist):
    '''
    Class to draw reproduction events in the dual of the SLFV with fixed radius
    and fixed impact parameter.
    '''
    def __init__(self, dim, params):
        '''
        Constructor

        Parameters
        ----------
        dim : int
            Dimension of the geographic space.
        params : dict
            Dictionary of parameters (easier to manipulate when inheriting from
            this class). For the fixed parameters version it should contain the
            keys 'impact' and 'radius', with numerical values.

        '''
        super().__init__(dim, params)
        assert params['impact'] > 0 and params['impact'] <= 1
        assert params['radius'] > 0
        self.u = params['impact']
        self.r = params['radius']
    
    def draw_event_params(self, lineage_position):
        '''
        Returns the parameters of a reproduction event, given that it involves 
        a lineage at the specified position (before thinning).

        Parameters
        ----------
        lineage_position : numpy.ndarray
            position of the lineage involved in the event (shape (d,)).

        Returns
        -------
        params: dict
            Dictionary containing the keys 'centre', 'radius', 'impact' and
            'parent position'.

        '''
        centre = uniform_draw_from_ball(lineage_position, self.r)
        parent_position = uniform_draw_from_ball(centre, self.r)
        params = {'centre': centre,
                  'radius': self.r,
                  'impact': self.u,
                  'parent position' : parent_position}
        return params

class TwoParentFixedRadius(DualFixedEvent):
    '''
    Class to draw reproduction events in the dual of the SLFV with two parents,
    with fixed impact and radius.
    '''
    def draw_event_params(self, lineage_position):
        params = super().draw_event_params(lineage_position)
        params['1st parent position'] = params['parent position']
        second_parent = uniform_draw_from_ball(params['centre'],
                                               params['radius'])
        params['2nd parent position'] = second_parent
        return params

class DualHeterogeneous(DualEventDist):
    '''
    Class to draw reproduction events in the dual of the SLFV with heterogeneous
    parameters (the impact and radius depend on the position of the centre of 
    the event relative to x_1 = 0).
    '''
    def __init__(self, dim, params):
        '''
        Constructor

        Parameters
        ----------
        dim : int
            dimension of geographical space (>= 1).
        params : dict
            Dictionary of paramters. Should contain the two impact paramters
            under the key 'impacts' (x_1 < 0 first, x_1 > 0 second), and the 
            two radii under 'radii'.

        '''
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
        # increase the jump rate of lineages who are close to the interface, since
        # they can be affected by both types of events.
        rates = np.ones(np.size(first_coords)) + (np.abs(first_coords) < self.R).astype(float)
        return rates
    
    def draw_event_params(self, lineage_position):
        # first pick a side for the centre of the reproduction event.
        if self.d > 1:
            x = lineage_position[0]
        else:
            x = lineage_position
        # p = 0 if the lineage is on the left, 0.5 if it is close to the interface
        # and 1 if it is on the right.
        p = 0.5 * (float(x > - self.R) + float(x > self.R))
        # a = 0 means that the event should be on the left, a = 1 means it should
        # be on the right
        a = np.random.binomial(1, p)
        centre = uniform_draw_from_ball(lineage_position, self.radii[a])
        # if the event is on the wrong side, we ignore it.
        if self.signs[a] * centre[0] < 0:
            raise IgnoreEvent()
        parent_position = uniform_draw_from_ball(centre, self.radii[a])
        params = {'centre': centre,
                  'radius': self.radii[a],
                  'impact': self.impacts[a],
                  'parent position': parent_position}
        return params

class TwoParentHeterogeneous(DualHeterogeneous, TwoParentFixedRadius):
    '''
    Class to draw reproduction events in the dual of the SLFV with two parents
    and with heterogeneous parameters.
    '''
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
        
