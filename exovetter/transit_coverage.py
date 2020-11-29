#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def calc_coverage(time, p_day, epoch, dur_hour, ndur=2, nbins=10):
    """Calculate the fraction of  the in-transit points that contain data.
    """
    phases = compute_phases(time, p_day, epoch, offset=0.5)

    phase_offset = ndur / 2 * dur_hour / 24.0 / p_day
    high = 0.5 + phase_offset
    low = 0.5 - phase_offset

    intransit = phases[(phases > low) & (phases <= high)]

    hist, bins = np.histogram(intransit, bins=nbins, range=(low, high))

    n_bins_with_data = np.sum(hist > 0)

    return n_bins_with_data / nbins, hist, bins


def compute_phases(time, period, epoch, offset=0.5):

    phases = np.fmod(time - epoch + (offset * period), period)

    return phases / period

def plot_coverage(hist,bins):
    
    plt.figure(figsize=(6,6))
    
    plt.step(bins[:-1], hist , color='blue', where="post", marker='.')
    
    plt.xlabel('Phase From Transit')
    plt.ylabel('Number of Data Points')
    plt.title('In Transit Coverage')
