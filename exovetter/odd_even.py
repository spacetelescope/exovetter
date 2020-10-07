#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#import trapezoid_fit as tzfit
#import matplotlib.pyplot as plt

def calc_odd_even(time, flux, period, epoch, duration, ingress = None):
    """

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.
    period : TYPE
        period in the same unit as time
    epoch : TYPE
        time of transit in same units as time
    durration : TYPE
        duration of transit in same units as time
    ingress : TYPE, optional
        ingress time in the same units as time

    Returns
    -------
    ratio : TYPE
        ratio between the odd and even depth
    sigma : TYPE
        significance of the ratio

    """
    offset = 0.25
    twicephase = compute_phases(time, 2*period, epoch, offset = offset)
    
    #plt.figure()
    #plt.plot(twicephase,flux, '.')
    
    dur_phase = duration/(2*period)
    
    odd_depth, even_depth = avg_odd_even(twicephase, flux, dur_phase, frac = 0.5, event_phase=offset)

    diff, error, sigma = calc_diff_significance(odd_depth, even_depth)
    
    return sigma, odd_depth, even_depth

def compute_phases(time, period, epoch, offset=0.25):

    phases = np.fmod(time - epoch + (offset * period), period)
    
    return phases/period    

def calc_ratio_significance(odd, even):
    
    error_ratio= (odd[0]/even[0]) * np.sqrt((odd[1]/odd[0])**2 + (even[1]/even[0])**2)
    ratio = odd[0]/even[0]
    
    sigma = (ratio-1)/error_ratio
    
    return ratio, sigma

def calc_diff_significance(odd, even):
    
    diff = np.abs(odd[0] - even[0])
    
    error = np.sqrt(odd[1]**2 + even[1]**2)
    
    if error != 0:
        sigma = diff/error
    else:
        sigma = np.nan
        
    return diff, error, sigma
            

def avg_odd_even(phases, flux, duration, event_phase=0.25, frac = 0.5):
    """
    This takes the phases when it is folded at twice the acutal period.
    The odds are considered to be at event_phase, evens are at event_phase+0.5.

    Parameters
    ----------
    phases : TYPE
        phases when folding at 2x the period.
    flux : TYPE
        relative flux of the light curve
    duration : TYPE
        duration of the transit in phase units
    event_phase : float
        phase of the odd transit
    frac : float
        fraction of the intransit points to use.

    Returns
    -------
    odd_depth: tuple of depth and error of the odd transit.
    even_depth: tuple of depth and error of the even transit.

    """
    
    odd_lower = event_phase - frac * 0.5 * duration
    odd_upper = event_phase + frac * 0.5 * duration
    
    even_lower = event_phase + 0.5 - frac * 0.5 * duration
    even_upper = event_phase + 0.5 + frac * 0.5 * duration
    
    even_transit_flux = flux[(phases > even_lower) & (phases < even_upper)]
    odd_transit_flux = flux[(phases > odd_lower) & (phases < odd_upper)]
    
    if (len(even_transit_flux) > 1) & (len(odd_transit_flux)>1):
        
        avg_even = np.average(even_transit_flux)
        avg_odd = np.average(odd_transit_flux)
        err_even = np.std(even_transit_flux)
        err_odd = np.std(odd_transit_flux)
        
        even_depth = (np.abs(avg_even), err_even)
        odd_depth = (np.abs(avg_odd), err_odd)
        
    else:
        even_depth = (1, 1)
        odd_depth = (1, 1)

    return odd_depth, even_depth

"""
import tce
import astropy.units as u


#Add these tests after new TCE class is created
def test_odd_even():
    

    times = np.arange(0, 400, .033)
    period = 100
    duration = 0.5
    epoch = 0.0
    noise = 0.009
    
    atce = tce.TCE(period * u.day, tzero = epoch, duration = duration * u.day,
               depth = 0.12, target_name='sample', event_name = "sample b")
    
    flux1 = atce.get_boxmodel(times)
    
    atce = tce.TCE(period*u.day, tzero=epoch+0.5*period, duration=duration*u.day,
               depth = 0.5, target_name='sample', event_name = "sample b")
    
    flux2 = atce.get_boxmodel(times) 
    
    flux = (flux1 + flux2)/2.0 + np.random.randn(len(flux1)) * noise
    
    sigma, odd, even = calc_odd_even(times, flux, period/2, epoch, duration)
    
    #plt.figure()
    #plt.plot(times, flux,'.')
    assert(sigma > 5)
    assert(odd[1] != 1)
    assert(even[1] != 1)


def test_odd_even2():
    

    times = np.arange(0, 400, .033)
    period = 100
    duration = 0.5
    epoch = 0.0
    noise = 0.1
    
    atce = tce.TCE(period * u.day, tzero = epoch, duration = duration * u.day,
               depth = 0.12, target_name='sample', event_name = "sample b")
    
    flux1 = atce.get_boxmodel(times)
    
    atce = tce.TCE(period*u.day, tzero=epoch+0.5*period, duration=duration*u.day,
               depth = 0.5, target_name='sample', event_name = "sample b")
    
    flux2 = atce.get_boxmodel(times) 
    
    flux = (flux1 + flux2)/2.0 + np.random.randn(len(flux1)) * noise
    plt.figure()
    plt.plot(times, flux,'.')
    
    sigma, odd, even = calc_odd_even(times, flux, period/2, epoch, duration)
    
    print(sigma, odd, even)
    assert(sigma < 3)
    assert(odd[1] != 1)
    assert(even[1] != 1)
"""

import pytest


def test_odd_even():
    #Simple test of folding
    t = np.arange(0,10)
    f = np.ones(len(t))
    f[1] = f[3] = f[5] = f[7] = f[9] = 0.5
    period = 2
    sigma, odd, even = calc_odd_even(t, f, period, epoch = 1, duration = 1)
    #print(sigma, odd, even)
    assert(odd[0] == 0.5)
    assert(even[0] == 0.5)
    assert(np.isnan(sigma))
    

def test_odd_even2():
    
    n = 100
    a = np.random.randn(n) + 2
    b = np.random.randn(n) + 10
    
    odd = (np.mean(a), np.std(a))
    even = (np.mean(b), np.std(b))
    #print(odd,even)
    diff, error, sigma = calc_diff_significance(odd, even)
    #print(diff, error, sigma)
    
    assert(diff == pytest.approx(8, abs=1))
    assert(error == pytest.approx(np.sqrt(2), rel = .1))
    assert(sigma == pytest.approx(5.6, abs=1))
    