#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:38:26 2022

Code to reproduce micheelle base-line vetting metrics.
from the class TransitLightCurve

"""

import numpy as np

from exovetter import utils

def transit_lc_masks(time, flux, error, period, epoch, 
                     duration, plot=False):
    
    """
    Calculate a set of masks for a light curve based on the 
    period and epoch of hte event.
    REturns to a dictionary arrays of the same length as time and flux.
    
    Parameters:
    --------
    time : float array
        time of the observations

    flux : float array
        relative flux normalized to zero to fit
        
    error : float array
        error in relative flux normalized to zero to fit

    period : float
        period in same units as time

    epoch : float
        time of transit event in same units as time and same offset

    duration : float
        transit event duration in units of time

    Returns
    -------
    result : dictionary
        dictionary of results with the following names:
        in_transit
        fit_intransit (2 transit durations on each side)
        right_intransit
        left_intransit
        odd_intransit
        even_intransit
        These can be useful for plotting and other things.
        
    """
    
    
    phases = utils.compute_phases(time, period, epoch, offset=0)
    qtran = duration/period
    
    in_transit = abs(phases) < 0.5*qtran
    fit_intransit = abs(phases) < 2 * qtran
    right_intransit = (phases > 0) & (phases > -0.5 * qtran)
    left_intransit = (phases < 0) & (phases > -0.5 * qtran)
    
    #phases for twice the period, for odd transits
    phases2 = utils.compute_phases(time, period*2, epoch, offset=0)   
    odd_intransit = abs(phases2) < 0.5 * qtran
    
    #even transits
    phases2 = utils.compute_phases(time,period*2, epoch+period, offset=0)
    even_intransit = abs(phases2) < 0.5 * qtran
    
    results = dict()
    results['in_transit'] = in_transit
    results['right_intransit'] = right_intransit
    results['left_intransit'] = left_intransit
    results['odd_intransit'] = odd_intransit
    results['even_intransit'] = even_intransit
    
    
    if plot:
        
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        
        ax[0,0].plot(phases, flux, "k.", label="Phased Flux")
        ax[0,0].plot(phases[in_transit], flux[in_transit], "C0.", label="1 transit duration")
        ax[0,0].plot(phases[fit_intransit], flux[fit_intransit], "C1.", label="4 transit durations")
        ax[0,0].set_xlim([-2.5*qtran, 2.5*qtran])
        ax[0,0].legend()
        
        ax[0,1].plot(time, flux, "k.", "Relative Flux")
        ax[0,1].plot(time[odd_intransit], flux[odd_intransit], "C0.", label="Odd transits")
        ax[0,1].plot(time[even_intransit],flux[even_intransit], "C1.", label="Even transits")
        ax[0,0].set_xlim(epoch, epoch + 6*period)
        ax[0,1].legend()
        
        ax[1,0].plot(phases, flux, "k.", "Relative Flux")
        ax[1,0].plot(phases[left_intransit], flux[left_intransit], "C0.", label="Left side of transits")
        ax[1,0].plot(phases[right_intransit], flux[right_intransit], "C1.", label="Right side of transits")
        ax[1,0].set_xlim([-2*qtran, 2*qtran])
        ax[1,0].legend()
        
        ax[1,1].plot(phases2, flux, "Phased Flux")
        ax[1,1].plot(phases2[odd_intransit], flux[odd_intransit], "C0.", label="Odd transits")
        ax[1,1].plot(phases2[even_intransit], flux[even_intransit], "C1.", label="Even transits")
        ax[1,1].set_xlim([-2*qtran, 2*qtran])
        ax[1,1].legend()
        
    return results

def get_SNR(y, dy, zpt, zpt_err):
    #Primary calculation for the MES
    #y is the flux in transit and dy is the error when fully in transit.
    #zpt and zpt_err is zero point and error we are working relative to.
    #Original code this was the avg for the near_transit (+-1 dur each side) 
    #weighted mean and error..
    
    avg, err = get_mean_and_error(y, dy)
    dep = zpt - avg
    err = np.sqrt(zpt_err**2 + err**2)
    
    mes = dep/err
    
    return mes

def get_SES_MES_series(time, flux, error, period, epoch, duration):
    """
    Given a flux time series and event,
    this returns the ses time series and mes time series

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.
    error : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.
    epoch : TYPE
        DESCRIPTION.
    duration : TYPE
        DESCRIPTION.

    Returns
    -------
    SES and MES time series
    Note, Max of MES time series is what is known as the MES of the event.

    """
    n = len(time)
    qtran = duration/period
    
    SES_series = np.zeros(n)
    MES_series = np.zeros(n)
    
    phases = utils.compute_phases(time, period, epoch, offset=0)
    phases[phases < 0] += 1
    
    mask_dict =  transit_lc_masks(time, flux, error, period, epoch, 
                     duration, plot=False)
    
    zpt, zpt_err = utils.calc_weighted_mean_and_error(flux[mask_dict['in_transit']], 
                                                      error[mask_dict['in_transit']])
    
    for i in range(n):
        # Get SES for this cadence - only use datapoints close in time
        in_tran = (abs(time - time[i]) < 0.5*duration)
        SES_series[i] = get_SNR(flux[in_tran], error[in_tran], zpt, zpt_err)
        
        # Get MES for this cadence - use all datapoints close in phase
        in_tran = (abs(phases - phases[i]) < 0.5*qtran) | (abs(phases - phases[i]) > 1-0.5*qtran)
        MES_series[i] = get_SNR(flux[in_tran], error[in_tran], zpt, zpt_err)

    SES_series = SES_series
    MES_series = MES_series