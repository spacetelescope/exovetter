# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:12:25 2020

@author: fergal
"""

import matplotlib.pyplot as plt
import numpy as np

import exovetter.utils as utils
import os

def sweet(time, flux, period, epoch, duration, plot=False):
    """Perform the SWEET test

    The SWEET test checks that the flux out of transit is not well
    fit by a sinecurve at either the proposed orbital period, or half,
    or twice that. It is a good check for both variable stars and
    ellopsoidal variation

    Inputs
    --------
    time, flux
        (1d numpy arrays) Data to fit
    period, epoch, duration
        (floats) Parameters of the transit.

    Returns
    -------
    A 2d numpy array. columns are amplitude, uncertainty and signal-to-noise.
    The rows are the fits at half the period, at the period, and twice the period.

    Note
    ------
    It is the caller's responsbility to ensure that the inputs are
    given in consistent units.
    """

    if len(time) != len(flux):
        raise ValueError('time and flux length mismatch')

    idx = np.isnan(time) | np.isnan(flux)
    time = time[~idx]
    flux = flux[~idx]

    flux -= np.mean(flux)

    idx = utils.mark_transit_cadences(time, period, epoch, duration)
    flux = flux[~idx]
    flux -= np.mean(flux)
    scatter = utils.estimate_scatter(flux)

    print("scatter is %f" %(scatter))

    if plot:
        plt.clf()

    out = []
    for i, per in enumerate([period * 0.5, period, 2 * period]):
        phase = np.fmod(time - epoch + per, per)
        phase = phase[~idx]
        period = np.max(phase)
        f_obj = utils.WqedLSF(phase, flux, scatter, period=period)
        amp, amp_unc = f_obj.compute_amplitude()
        out.append([amp, amp_unc, amp / amp_unc])

        if plot:
            srt = np.argsort(phase)
            plt.subplot(3,1,i+1)
            plt.plot(phase, flux, 'ko')
            plt.plot(phase[srt], f_obj.get_best_fit_model(phase[srt]), '-')
            plt.ylabel("P=%g" %(per))

    result = np.array(out)

    return result


def construct_message(result, threshold_sigma):
    msg = []
    if result[0, -1] > threshold_sigma:
        msg.append("WARN: SWEET test finds signal at HALF transit period")
    if result[1, -1] > threshold_sigma:
        msg.append("WARN: SWEET test finds signal at the transit period")
    if result[2, -1] > threshold_sigma:
        msg.append("WARN: SWEET test finds signal at TWICE the "
                    "transit period")
    if len(msg) == 0:
        msg = [("OK: SWEET finds no out-of-transit variability at "
                "transit period")]

    return {'msg': os.linesep.join(msg), 'amp': result}


