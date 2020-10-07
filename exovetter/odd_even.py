#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt


def calc_odd_even(time, flux, period, epoch, duration, ingress=None):
    """

    Parameters
    ----------
    time : array
        times
    flux : array
        relative flux normalized to 1
    period : float
        period in the same unit as time
    epoch : float
        time of transit in same units as time
    durration : float
        duration of transit in same units as time
    ingress : float, optional
        ingress time in the same units as time
        currently unused

    Returns
    -------
    diff : float
        difference between the odd and even depth
    error : float
        error on the
    sigma : float
        significance that the difference is not zero
    """

    offset = 0.25
    twicephase = compute_phases(time, 2 * period, epoch, offset=offset)

    # plt.figure()
    # plt.plot(twicephase,flux, '.')

    dur_phase = duration / (2 * period)

    odd_depth, even_depth = avg_odd_even(
        twicephase, flux, dur_phase, frac=0.5, event_phase=offset)

    diff, error, sigma = calc_diff_significance(odd_depth, even_depth)

    return sigma, odd_depth, even_depth


def compute_phases(time, period, epoch, offset=0.25):

    phases = np.fmod(time - epoch + (offset * period), period)

    return phases / period


def calc_ratio_significance(odd, even):

    error_ratio = (odd[0] / even[0]) * \
        np.sqrt((odd[1] / odd[0])**2 + (even[1] / even[0])**2)
    ratio = odd[0] / even[0]

    sigma = (ratio - 1) / error_ratio

    return ratio, sigma


def calc_diff_significance(odd, even):

    diff = np.abs(odd[0] - even[0])

    error = np.sqrt(odd[1]**2 + even[1]**2)

    if error != 0:
        sigma = diff / error
    else:
        sigma = np.nan

    return diff, error, sigma


def avg_odd_even(phases, flux, duration, event_phase=0.25, frac=0.5):
    """
    This takes the phases when it is folded at twice the acutal period.
    The odds are considered to be at event_phase, evens are at event_phase+0.5.

    Parameters
    ----------
    phases : array
        phases when folding at 2x the period.
    flux : array
        relative flux of the light curve
    duration : float
        duration of the transit in same units as phase
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

    if (len(even_transit_flux) > 1) & (len(odd_transit_flux) > 1):

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
