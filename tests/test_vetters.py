#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:16:20 2020
"""

import pytest
from numpy.testing import assert_allclose
from astropy.io import ascii
from astropy import units as u
import lightkurve as lk

from exovetter import exo_const
from exovetter import vetters
from exovetter.tce import Tce


def wasp18():
    d = {'period':  0.94124 * u.day,
         'epoch': 58374.669883 * u.day,
         'epoch_offset':  -2400000.5 * u.day,
         'depth': 0.00990112 * exo_const.frac_amp,
         'duration': 0.08932 *u.day,
         'event_name': 'WASP-18 b',
         'target_name': 'WASP-18'}
    
    return d

def get_lightcurve():
    
    lc_file = "./wasp18b_flat_lightcurve.csv"
    
    lc_table = ascii.read(lc_file, data_start=1)
    
    lc = lk.LightCurve(time=lc_table[1], flux=lc_table[2], 
                       flux_err=lc_table[3])
    
    return lc
    