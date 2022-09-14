#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:53:09 2022

@author: smullally
"""

import numpy as np

from exovetter.centroid import centroid as cent
from exovetter import const as exo_const
from exovetter import vetters
from exovetter.tce import Tce

from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.io import ascii
import lightkurve as lk


def get_toi175_tce():

    tce = Tce(
        period=3.6906297716078114 * u.day,
        epoch=1356.2124667131718 * u.day,
        epoch_offset=exo_const.btjd,
        depth=0.00001315 * exo_const.frac_amp,
        duration=0.0642 * u.day,
        event_name="TOI-175",
        target_name="TIC 307210830",
        snr=20,
    )

    return tce


def get_toi175_lightcurve():

    lc_file = get_pkg_data_filename("data/tic307210830_eleanor_lightcurve.csv")

    lc_table = ascii.read(lc_file, data_start=1)

    lc = lk.LightCurve(
        time=lc_table["time"],
        flux=lc_table["flux"],
        flux_err=lc_table["err"],
        time_format="btjd",
        cadenceno = lc_table["cadence"]
    )
    lc.add_column(lc_table['raw'],name="raw")
    
    
    return lc

def test_vetter_uni():
    
    lc = get_toi175_lightcurve()
    rolc = lc.remove_outliers(sigma=10)
    tce = get_toi175_tce()
    
    tessvet = vetters.TessTransitEventStats(error_name="flux_err")
    result, tcelc = tessvet.run(tce, rolc)
    
    assert np.allclose(result['uni_sig_pri'], 82.68, atol=10)
    assert np.allclose(result['uni_shape'], 0.04, atol=10)
    assert np.allclose(result['Ntransits'], 85, atol=.1)