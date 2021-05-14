# Code to plot and evaluate individual transits.
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel
from exovetter import utils


def plot_all_transits(time, flux, period, epoch, dur, depth, max_transits=20,
                      transit_only=False, plot=True):
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

    Returns
    -------
    n_has_data = int
        Number of transits with data in transit (3*duration)

    """

    phases = utils.compute_phases(time, period, epoch, offset=0.25)
    intransit = utils.mark_transit_cadences(time, period, epoch, dur,
                                            num_durations=3, flags=None)

    xmin = 0
    xmax = np.max(phases) * period
    figwid = 8
    if transit_only:
        xmin = np.min(phases[intransit]) * period
        xmax = np.max(phases[intransit]) * period

        if (xmax - xmin) > (0.5 * period):
            xmin = (0.25 * period) - 1.25 * dur
            xmax = (0.25 * period) + 1.25 * dur

        figwid = 4

    offset = 0.25
    ntransit = np.floor((time - epoch + (offset * period)) / period)

    n_has_data = len(np.unique(ntransit[intransit]))

    #step_size = 6*np.std(flux[~intransit])
    step_size = depth

    nsteps = int(np.max(ntransit))

    if nsteps > max_transits:
        nsteps = max_transits

    if plot:
        plt.figure(figsize=(figwid, nsteps))
        for nt in np.arange(0, nsteps, 1):
            ph = phases[ntransit == nt]
            fl = flux[ntransit == nt]

            color = (0, 0.3 - 0.3 * (nt / nsteps), nt / nsteps)

            plt.plot(
                ph *
                period,
                fl +
                step_size *
                nt,
                '.--',
                c=color,
                ms=5,
                lw=1)
            plt.annotate("Transit %i" % nt, (xmin, np.median(fl) + step_size * nt),
                         c=color)

        plt.xlim(xmin, xmax)
        plt.xlabel("Phased Time")

    return n_has_data


def plot_fold_transit(time, flux, period, epoch, depth, dur, smooth=10,
                      transit_only=False, plot=True):
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
    smooth : integer, optional
        Approximately number of points you want across 3 in-transit durations
        for a
        1DBoxkernel. The default is 10. None will turn off smoothing.

    Returns
    -------
    None.

    """

    phases = utils.compute_phases(time, period, epoch, offset=0.25)

    intransit = utils.mark_transit_cadences(time, period, epoch, dur,
                                            num_durations=3, flags=None)

    if smooth is not None:
        N = int(np.floor(len(phases[intransit]) / smooth))
        sort_index = np.argsort(phases)
        smoothed_signal = convolve(flux[sort_index], Box1DKernel(N))

    if plot:
        plt.figure(figsize=(8, 6))

        plt.plot(phases * period, flux, 'k.', ms=3, label="Folded")

        if smooth is not None:
            sort_phases = phases[sort_index]
            plt.plot(sort_phases[N:-N] * period, smoothed_signal[N:-N], 'r--',
                     lw=1.5, label="Box1DSmooth")

        plt.legend(loc="upper right")
        plt.xlabel('Phased Times')

        if transit_only:
            xmin = np.min(phases[intransit]) * period
            xmax = np.max(phases[intransit]) * period
            plt.xlim(xmin, xmax)
