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

from exovetter import const as exo_const
from exovetter import vetters
from exovetter.tce import Tce


def get_wasp18_tce():
    
    tce = Tce(period = 0.94124 * u.day,
         epoch=58374.669883 * u.day,
         epoch_offset= -2400000.5 * u.day,
         depth=0.00990112 * exo_const.frac_amp,
         duration=0.08932 *u.day,
         event_name= 'WASP-18 b',
         target_name= 'WASP-18',
         snr = 50)
    
    return tce

def get_wasp18_lightcurve():
    
    lc_file = "./wasp18b_flat_lightcurve.csv"
    
    lc_table = ascii.read(lc_file, data_start=1)
    
    lc = lk.LightCurve(time=lc_table['col2'], flux=lc_table['col3'], 
                       flux_err=lc_table['col4'], time_format="btjd")
    
    return lc

def test_vet():
    
    tce = get_wasp18_tce()
    lc = get_wasp18_lightcurve()
    print(lc.time)
    
    metrics = dict()
    vetter_list = [vetters.Lpp(), 
                   vetters.OddEven(),
                   vetters.TransitPhaseCoverage()]
    
    for v in vetter_list:
        vetter = v
        _ = vetter.run(tce, lc)
        metrics.update(vetter.__dict__)
                       
    assert_allclose(metrics['norm_lpp'], 7.93119, rtol=1e-3)

    
    # def run_many_vetters(tce, vetterlist, **kwargs):
#     import lightkurve
#     lc = lightkurve.load_lightcurve_tess(tce, **kwargs)

#     metrics = dict()
#     for v in vetterlist:
#         metrics.update(v.run(tce, lc))

#     return metrics