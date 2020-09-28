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
"""
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
"""

def test_modshift():
    np.random.seed(1234)
    x = np.arange(100) + .25
    y = np.random.randn(100)
    y[40:50] -= 5  #Primary
    y[80:90] -= 3 #secondary

    model = np.zeros(len(x))
    model[40:48] = -5

    period_days = len(x)
    epoch = 44
    duration_hrs = 10*24

    res = modshift.compute_modshift_metrics(x, y, model, period_days, epoch, duration_hrs)

    assert np.isclose(res['pri'], 0, atol=2), res
    assert np.isclose(res['sec'], 84-44, atol=2), res

    # plt.clf()
    # plt.plot(x, y, 'k.')
    return res


def test_fold_and_bin_data():
    x = np.arange(100)+ .25
    y = np.random.randn(100)

    data = modshift.fold_and_bin_data(x, y, 100, 0, 500, .25)
    assert len(data) == len(x), len(data)

    data = modshift.fold_and_bin_data(x, y, 100, 0, 500, .25)
    assert len(data) == len(x), len(data)

    data = modshift.fold_and_bin_data(x, y, 100, 44, 500, .25)
    assert len(data) == len(x), len(data)

    # idebug()
    data = modshift.fold_and_bin_data(x, y, 100, 44, 100, .25)
    assert len(data) == len(x), len(data)


def test_convolve():
    size = 24
    phase = np.arange(size)
    flux = np.zeros(size)
    flux[0] = -1
    flux[-1] = -1
    flux[8:10] = -.5

    model = np.zeros(size)
    model[0] = -1
    model[-1] = -1

    conv = modshift.compute_convolution_for_binned_data(phase, flux, model)
    assert len(conv) == len(phase), len(conv)

    print(conv)
    assert np.isclose(conv[0],  -2), conv[0]
    assert np.isclose(conv[9],  -1, atol=1e-3), conv[9]
