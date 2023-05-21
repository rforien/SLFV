#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:14:46 2021

@author: raphael
"""

__all__ = ['slfv', 'SLFV_dual', 'function', 'events', 'frequency', 'ARG']


from .slfv import SLFV
from .SLFV_dual import SLFV_dual, SLFV_ARG
from . import events
from . import ARG
# from .slfv import SLFV
# from .SLFV_dual import SLFV_dual, SLFV_ARG
# from .event_drawer import *
# # from .frequency import *