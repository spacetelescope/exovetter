# Code to plot and evaluate individual transits.
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel
from exovetter import utils


def plot_all_transits(time, flux, period, epoch, dur, depth, max_transits=20,
                      transit_only=False, plot=True, units="d"):
    """

    Parameters
    ----------
    time : numpy array
        times of measurements
    flux : numpy array
        brightness changes
    period : float
        period in same units as time
    epoch : float
        epoch of transits in same units and offset as time.
    dur : float
        duration of the transit in same units as the time.
    depth : float
        depth of possible transit, used to spread out each light curve.
    max_transits : integer, optional
        maximum number of transits to plot. The default is 10.
    transit_only : T/F
        if true, zooms the x axis around just 3 durations around the transit
    plot : boolean
        if true, it shows the plot, if false it does not
    units : string
        default is 'd'. Time Units to put on the plot.

    Returns
    -------
    n_has_data = int
        Number of transits with data in transit (3*duration)

    """

    phases = utils.compute_phases(time, period, epoch, offset=0.25)
    intransit = utils.mark_transit_cadences(time, period, epoch, dur,
                                            num_durations=3, flags=None)

    xmin = 0
    xmax = np.max(phases)
    figwid = 8
    if transit_only:
        xmin = np.min(phases[intransit])
        xmax = np.max(phases[intransit])

        if (xmax - xmin) > (0.5 * period):
            xmin = (0.25 * period) - 1.25 * dur
            xmax = (0.25 * period) + 1.25 * dur

        figwid = 4

    offset = 0.25
    ntransit = np.floor((time - epoch + (offset * period)) / period)

    #pmin = np.min(ntransit)
    #n = np.ceil(np.abs(pmin / period))
    #ntransit[ntransit < 0] = ntransit[ntransit < 0] + (period * n)

    n_has_data = len(np.unique(ntransit[intransit]))

    #step_size = 6*np.std(flux[~intransit])
    step_size = depth * 1.25
    if 3 * np.std(flux[~intransit]) > step_size:
        step_size = 3 * np.std(flux[~intransit])

    nsteps = len(np.unique(ntransit))  #np.ceil(np.max(ntransit))

    if nsteps > max_transits:
        nsteps = max_transits

    if plot:
        plt.figure(figsize=(figwid, nsteps))
        transits  =  np.floor(np.unique(ntransit[intransit]))
        print(transits[0:nsteps])
        print(nsteps)
        print(ntransit)
        
        for i, nt in enumerate(transits[0:nsteps]): 
            
            ph = phases[ntransit == nt]
            fl = flux[ntransit == nt]

            color = (0, 0.3 - 0.3 * (i / nsteps), i / nsteps)

            plt.plot(ph, fl + step_size * i, '.--',
                     c=color, ms=5, lw=1)
            plt.annotate("Transit %i" % nt,
                         (xmin, np.median(fl) + step_size * i),
                         c=color)

        plt.xlim(xmin, xmax)
        plt.axvspan(
            offset * period - 0.5 * dur,
            offset * period + 0.5 * dur,
            alpha=0.15)
        plt.xlabel("Phased Time [%s]" % units)

    return n_has_data


def plot_fold_transit(time, flux, period, epoch, depth, dur, smooth=10,
                      transit_only=False, plot=True, units="d"):
    """
    Bins set to None will show not show the binned points. Otherwise
    the binning is chosen

    Parameters
    ----------
    time : numpy array
        times of measurements
    flux : numpy array
        brightness changes
    period : float
        period in same units as time
    epoch : float
        epoch of transits in same units and offset as time.
    dur : float
        duration of the transit in same units as the time.
    transit_only: True/False
        Center the plot around the transit, only showing 3 durations
    smooth : integer, optional
        Approximately number of points you want across 3 in-transit durations
        for a
        1DBoxkernel. The default is 10. None will turn off smoothing.
    units : str, default="d"
        Unit string to put in the plot x-axis for clarity.

    Returns
    -------
    None.

    """
    offset = 0.25
    phases = utils.compute_phases(time, period, epoch, offset=offset)

    intransit = utils.mark_transit_cadences(time, period, epoch, dur,
                                            num_durations=3, flags=None)

    if smooth is not None:
        N = int(np.floor(len(phases[intransit]) / smooth))
        sort_index = np.argsort(phases)
        smoothed_signal = convolve(flux[sort_index], Box1DKernel(N))

    if plot:
        plt.figure(figsize=(8, 6))

        plt.plot(phases, flux, 'k.', ms=3, label="Folded")
        plt.axvspan(
            offset *
            period -
            0.5 *
            dur,
            offset *
            period +
            0.5 *
            dur,
            alpha=0.15)

        if smooth is not None:
            sort_phases = phases[sort_index]
            plt.plot(sort_phases[N:-N], smoothed_signal[N:-N], 'r--',
                     lw=1.5, label="Box1DSmooth")

        plt.legend(loc="upper right")
        plt.xlabel('Phased Times [%s]' % units)

        if transit_only:
            xmin = np.min(phases[intransit])
            xmax = np.max(phases[intransit])
            plt.xlim(xmin, xmax)
