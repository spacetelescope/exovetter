# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:24:11 2020

@author: fergal
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import astropy.io.fits as pyfits
import exovetter.modshift.modshift as modshift

def prep_test_data():
    fn = "/home/fergal/data/kepler/Q6/1026/kplr001026032-2010265121752_llc.fits"
    data, hdr = pyfits.getdata(fn, header=True)

    time = data['TIME']
    flux = data['PDCSAP_FLUX']
    idx = np.isfinite(time) & np.isfinite(flux)
    time = time[idx]
    flux = flux[idx]

    flux /= np.median(flux)
    flux -= 1

    data = np.vstack([time, flux]).transpose()
    np.savez('modshift_test_data.npz', data=data)


def test_modshift():
    data = np.load('modshift_test_data.npz')['data']
    time = data[:,0]
    flux = data[:,1]

    tce = dict()
    tce[modshift.names.PERIOD] = 8.4604378
    tce[modshift.names.EPOCH] = 54966.77813 - 54832.0
    tce[modshift.names.DEPTH] = .07 * 1e6
    tce[modshift.names.DURATION_HRS] = 3

    vals = modshift.compute_modshift_metrics(time, flux, tce, modshift.box_car)
    assert vals['sigma_pri'] < -200
    assert vals['sigma_sec'] < -90
    assert vals['sigma_ter'] < -1
    assert vals['sigma_pos'] > 2
