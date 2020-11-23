"""Simple average-based odd/even vetter."""
import numpy as np

from exovetter.transit_coverage import compute_phases

__all__ = ['calc_odd_even', 'calc_ratio_significance',
           'calc_diff_significance', 'avg_odd_even']


def calc_odd_even(time, flux, period, epoch, duration, ingress=None):
    """Simple odd/even vetter.

    Parameters
    ----------
    time : array
        Times.

    flux : array
        Relative flux normalized to 1.

    period : float
        Period in the same unit as time.

    epoch : float
        Time of transit in same units as time.

    duration : float
        Duration of transit in same units as time.

    ingress : float, optional
        Ingress time in the same units as time.
        **This keyword is currently unused.**

    Returns
    -------
    sigma : float
        Significance that the difference is not zero.

    odd_depth : float
        Odd depth.

    even_depth : float
        Even depth.

    """
    offset = 0.25
    twicephase = compute_phases(time, 2 * period, epoch, offset=offset)

    dur_phase = duration / (2 * period)

    odd_depth, even_depth = avg_odd_even(
        twicephase, flux, dur_phase, frac=0.5, event_phase=offset)

    diff, error, sigma = calc_diff_significance(odd_depth, even_depth)

    return sigma, odd_depth, even_depth


def calc_ratio_significance(odd, even):
    """Calculate ratio significance between odd and even.

    Parameters
    ----------
    odd, even : float

    Returns
    -------
    ratio : float
        Ratio of odd to even.

    sigma : float
        Significance that the ratio is not 1.

    """
    error_ratio = ((odd[0] / even[0]) *
                   np.sqrt((odd[1] / odd[0])**2 + (even[1] / even[0])**2))
    ratio = odd[0] / even[0]

    sigma = (ratio - 1) / error_ratio

    return ratio, sigma


def calc_diff_significance(odd, even):
    """Calculate difference significance between odd and even.

    Parameters
    ----------
    odd, even : float

    Returns
    -------
    diff : float
        Difference of ``odd - even``.

    error : float
        Uncertainty of the difference.

    sigma : float
        Significance that the difference is not zero.

    """
    diff = np.abs(odd[0] - even[0])
    error = np.sqrt(odd[1]**2 + even[1]**2)

    if error != 0:
        sigma = diff / error
    else:
        sigma = np.nan

    return diff, error, sigma


def avg_odd_even(phases, flux, duration, event_phase=0.25, frac=0.5):
    """Simple average-based odd/even vetter.

    This takes the phases when it is folded at twice the acutal period.
    The odds are considered to be at ``event_phase``,
    evens are at ``event_phase + 0.5``.

    Parameters
    ----------
    phases : array
        Phases when folding at 2x the period.

    flux : array
        Relative flux of the light curve.

    duration : float
        Duration of the transit in same units as phases.

    event_phase : float
        Phase of the odd transit.

    frac : float
        Fraction of the in-transit points to use.

    Returns
    -------
    odd_depth : tuple of float
        Depth and error of the odd transit.

    even_depth : tuple of float
        Depth and error of the even transit.

    """
    x = frac * 0.5 * duration

    odd_lower = event_phase - x
    odd_upper = event_phase + x

    even_lower = odd_lower + 0.5
    even_upper = odd_upper + 0.5

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
