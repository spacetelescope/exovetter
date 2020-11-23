"""Module to handle transit coverage calculations."""
import numpy as np

__all__ = ['calc_coverage', 'compute_phases']


# TODO: Improve docstring.
def calc_coverage(time, p_day, epoch, dur_hour, ndur=2, nbins=10):
    """Calculate the fraction of the in-transit points that contain data.

    Parameters
    ----------
    time, p_day, epoch : float
        See :func:`compute_phases`.

    dur_hour : float
        Duration in hours.

    ndur : int
        Number of duration.

    nbins : int
        Number of bins.

    Returns
    -------
    coverage : float
        Fraction of the in-transit points that contain data.

    """
    phases = compute_phases(time, p_day, epoch, offset=0.5)

    phase_offset = ndur / 2 * dur_hour / 24.0 / p_day
    high = 0.5 + phase_offset
    low = 0.5 - phase_offset

    intransit = phases[(phases > low) & (phases <= high)]

    hist, bins = np.histogram(intransit, bins=nbins, range=(low, high))

    n_bins_with_data = np.sum(hist > 0)

    return n_bins_with_data / nbins


# TODO: Improve docstring.
def compute_phases(time, period, epoch, offset=0.5):
    """Calculate phases.

    Parameters
    ----------
    time : float
        Time.

    period : float
        Period in days.

    epoch : float
        Epoch.

    offset : float
        Offset.

    Returns
    -------
    phases : float
        Phases.

    """
    phases = np.fmod(time - epoch + (offset * period), period)
    return phases / period
