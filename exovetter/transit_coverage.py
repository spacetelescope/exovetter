"""Module to handle transit coverage calculations."""
import numpy as np

__all__ = ['calc_coverage', 'compute_phases']


def calc_coverage(time, p_day, epoch, dur_hour, ndur=2, nbins=10):
    """Calculate the fraction of the in-transit points that contain data.

    Parameters
    ----------
    time, p_day, epoch : float
        See :func:`compute_phases`.

    dur_hour : float
        Duration in hours.

    ndur : int
        Multiplicative factor to specify the length of the in-transit points
        for the calculation. 1 = only actual in transit points.

    nbins : int
        Number of bins to split up the in transit points.

    Returns
    -------
    coverage : float
        Fraction of the in-transit points that contain data.

    hist : array
        Histogram of the times of length nbins

    bins : array
        corners of the bins for the histogram, length of nbins+1

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
    """Calculate phases.

    Parameters
    ----------
    time : float
        Time

    period : float
        Period in units of time.

    epoch : float
        Time of the transit in units of the time.

    offset : float
        Fractional value the times of the epoch should land on.

    Returns
    -------
    phases : float
        Fractional phases of the times given the period, epoch and offset.

    """
    phases = np.fmod(time - epoch + (offset * period), period)
    return phases / period


def plot_coverage(hist, bins):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))

    plt.step(bins[:-1], hist, color='blue', where="post", marker='.')

    plt.xlabel('Phase From Transit')
    plt.ylabel('Number of Data Points')
    plt.title('In Transit Coverage')
