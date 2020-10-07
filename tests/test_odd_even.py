#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import exovetter.odd_even as oe

def test_odd_even():
    #Simple test of folding with no odd even and no error
    t = np.arange(0,10)
    f = np.ones(len(t))
    f[1] = f[3] = f[5] = f[7] = f[9] = 0.5
    period = 2
    sigma, odd, even = oe.calc_odd_even(t, f, period, epoch = 1, duration = 1)
    
    assert(odd[0] == 0.5)
    assert(even[0] == 0.5)
    assert(np.isnan(sigma))


def test_odd_even2():
    
    n = 100
    a = np.random.randn(n) + 2
    b = np.random.randn(n) + 10
    
    odd = (np.mean(a), np.std(a))
    even = (np.mean(b), np.std(b))

    diff, error, sigma = oe.calc_diff_significance(odd, even)
    
    assert(diff == pytest.approx(8, abs=1))
    assert(error == pytest.approx(np.sqrt(2), rel = .1))
    assert(sigma == pytest.approx(5.6, abs=1))
    

#Add these tests after the new TCE class is settled upon.  
"""
import tce
import astropy.units as u

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