# -*- coding: utf-8 -*-
"""
Compute Jeff Coughlin's Modshift metrics.

Coughlin uses modshift to refer to both the transit significance tests, as
well as a suite of other, related, tests. This code only measures the
metrics for the transit significance measurements. So, for example,
the Odd Even test is not included here.

The algorithm is as follows:

* Fold and bin the data
* Convolve binned data with model.
* Identify the three strongest dips, and the strongest poxsitive excursion
* Remove some of these events, and measure scatter of the rest of the data
* Scale the convolved data by the per-point scatter so that each point
  represents the statistical significance of a transit signal at that phase.
* Record the statistical significance of the 4 events.

"""

import numpy as np
from scipy import special as spspec
from scipy import integrate as spint

import exovetter.modshift.plotmodshift as plotmodshift

__all__ = ['compute_modshift_metrics', 'fold_and_bin_data',
           'compute_false_alarm_threshold', 'compute_event_significances',
           'find_indices_of_key_locations', 'mark_phase_range',
           'estimate_scatter', 'compute_convolution_for_binned_data',
           'compute_phase']


def compute_modshift_metrics(time, flux, model, period_days, epoch_days,
                             duration_hrs, show_plot=True):
    """Compute Jeff Coughlin's Modshift metrics.

    This algorithm is adapted from the Modshift code used in the Kepler
    Robovetter and documented on
    https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19105-002.pdf

    (see page 30, and appropriate context)

    Jeff uses modshift to refer to both the transit significance tests, as
    well as a suite of other, related, tests. This code only measures the
    metrics for the transit significance measurements.

    The algorithm is as follows:

    * Fold and bin the data
    * Convolve binned data with model
    * Identify the three strongest dips, and the strongest positive excursion
    * Remove some of these events, and measure scatter of the rest of the data
    * Scale the convolved data by the per-point scatter so that each point
      represents the statistical significance of a transit signal at that
      phase.
    * Record the statistical significance of the 4 events.

    Parameters
    ----------
    time
        (1d numpy array) times of observations in units of days
    flux
        (1d numpy array) flux values at each time. Flux should be in
        fractional amplitude (with typical out-of-transit values close to zero)
    model
        (1d numpy array) Model transit flux based on the properties of the TCE
        len(model) == len(time)
    period_days, epoch_days, duration_hrs : float
        Properties of the transit
    show_plot : bool
        Display plot. This needs ``matplotlib`` to be installed.

    Returns
    -------
    results : dict

    Raises
    ------
    ValueError
        Invalid inputs.
    """
    if not np.all(np.isfinite(time)):
        raise ValueError('time must contain all finite values')
    if not np.all(np.isfinite(flux)):
        raise ValueError('flux must contain all finite values')
    if len(time) != len(flux):
        raise ValueError('time and flux must be of same length')

    overres = 10  # Number of bins per transit duration

    numBins = overres * period_days * 24 / duration_hrs
    numBins = int(numBins)

    data = fold_and_bin_data(time, flux, period_days, epoch_days, numBins)
    bphase = data[:, 0]
    bflux = data[:, 1]

    # Fold the model here
    bModel = fold_and_bin_data(time, model, period_days, epoch_days, numBins)
    bModel = bModel[:, 1] / bModel[:, 2]  # Avg flux per bin

    # Scale model so integral from 0.. period is 1
    integral = -1 * spint.trapz(bModel, bphase)
    bModel /= integral

    conv = compute_convolution_for_binned_data(bphase, bflux, bModel)
    if len(conv) != len(bphase):
        raise ValueError('conv and bphase must be of same length')
    results = find_indices_of_key_locations(bphase, conv, duration_hrs)

    phi_days = compute_phase(time, period_days, epoch_days)
    sigma = estimate_scatter(
        phi_days, flux, results["pri"], results["sec"], 2 * duration_hrs)
    results.update(compute_event_significances(conv, sigma, results))

    results["false_alarm_threshold"] = compute_false_alarm_threshold(
        period_days, duration_hrs)

    results["Fred"] = np.std(conv) / np.std(bflux)  # orig code had np.nan

    if show_plot:
        plotmodshift.plot_modshift(phi_days, period_days, flux, model,
                                   conv, results)
    return results, conv


def fold_and_bin_data(time, flux, period, epoch, num_bins):
    """Fold data, then bin it up.

    Parameters
    ----------
    time
        (1d numpy array) times of observations
    flux
        (1d numpy array) flux values at each time. Flux should be in
        fractional amplitude (with typical out-of-transit values close
        to zero)
    period : float
        Orbital Period of TCE. Should be in same units as *time*
    epoch : float
        Time of first transit of TCE. Same units as *time*
    num_bins : int
        How many bins to use for folded, binned, lightcurve

    Returns
    -------
    out
        2d numpy array. columns are phase (running from 0 to period),
        binned flux, and counts. The binned flux is the sum of all fluxes
        in that bin, while counts is the number of flux points added to the
        bin. To plot the binned lightcurve, you will want to find the average
        flux per bin by dividing binnedFlux / counts. Separating out the
        two components of average flux makes computing the significance
        of the transit easier.

    Notes
    -----
    This isn't everything I want it to be. It assumes that every
    element in y, falls entirely in one bin element, which is not
    necessarily true.
    """
    i = np.arange(num_bins + 1)
    bins = i / float(num_bins) * period  # 0..period in numBin steps

    phase = compute_phase(time, period, epoch)
    srt = np.argsort(phase)
    phase = phase[srt]
    flux = flux[srt]

    cts = np.histogram(phase, bins=bins)[0]
    binnedFlux = np.histogram(phase, bins=bins, weights=flux)[0]
    idx = cts > 0

    numNonZeroBins = np.sum(idx)
    out = np.zeros((numNonZeroBins, 3))
    out[:, 0] = bins[:-1][idx]
    out[:, 1] = binnedFlux[idx]
    out[:, 2] = cts[idx]
    return out


def compute_false_alarm_threshold(period_days, duration_hrs):
    """Compute the stat, significance needed to invalidate the null hypothesis

    An event should be considered statistically significant if its
    peak in the convolved lightcurves is greater than the value computed
    by this function.

    Note that this number is computed on a per-TCE basis. If you are looking
    at many TCEs you will need a stronger threshold. (For example, if
    you use this function to conclude that there is a less than 0.1% chance
    a given event is a false alarm due to Gaussian noise, you expect to
    see one such false alarm in 1,000 TCEs. See Coughlin et al. for the
    formula to ensure less than 1 false alarm over many TCEs.

    Parameters
    ----------
    period_days : float
        Orbital period
    duration_hrs : float
        Duration of transit in hours.

    Returns
    -------
    fa : float
        **TODO** What exactly is returned. Is this the 1 sigma false
        alarm threshold?
    """
    duration_days = duration_hrs / 24.0

    fa = spspec.erfcinv(duration_days / period_days)
    fa *= np.sqrt(2)
    return fa


def compute_event_significances(conv, sigma, results):
    """Compute the statistical significance of 4 major events

    The 4 events are the primary and secondary transits, the "tertiary
    transit", i.e the 3rd most significant dip, and the strongest postive
    signal.

    These statistical significances are the 4 major computed metrics of
    the modshift test.

    Parameters
    ----------
    conv
        (2d np array)  The convolution of the folded lightcurve and the
        transit model As computed by `compute_convolution_for_binned_data`
    sigma : float
        As returned by `estimate_scatter`

    results : dict
        Contains the indices in ``conv`` of the 4 events. These indices
        are stored in the keys "pri", "sec", "ter", "pos"

    Returns
    -------
    out : dict
        The ``results`` dictionary is returned, with 4 additional keys added,
        'sigma_pri', 'sigma_sec', etc. These contain the statistical
        significances of the 4 major events.

    Raises
    ------
    ValueError
        Invalid inputs.
    """

    if sigma <= 0:
        raise ValueError('sigma must be positive')

    event_keys = ["pri", "sec", "ter", "pos"]

    if not (set(event_keys) <= set(results.keys())):
        raise ValueError(f'results must contains these keys: {event_keys}')

    conv = conv.copy() / sigma  # conv is now in units of statistical signif
    out = dict()
    for key in event_keys:
        i0 = results[key]
        outKey = f"sigma_{key}"

        if np.isnan(i0):
            out[outKey] = np.nan
        else:
            out[outKey] = conv[i0]
    return out


def find_indices_of_key_locations(phase, conv, duration_hrs):
    """Find the locations of the 4 major events in the convolved data.

    The 4 major events are the primary transit, the secondary transit,
    the tertiary transit (i.e the 3rd most significant dip), and the
    most postitive event. This function finds their location in the
    folded (and binned) lightcurve convolved with the transit model.

    Parameters
    ----------
    conv
        (2d np array)
        See output of `compute_convolution_for_binned_data`
    period_days, duration_hrs : float

    Returns
    -------
    out : dict
        Each value is an index into the conv array.
    """
    conv = conv.copy()
    out = dict()
    transit_width = duration_hrs / 24.0
    gap_width = 2 * transit_width
    pos_gap_width = 3 * transit_width

    i0 = int(np.argmin(conv))
    out["pri"] = i0
    out["phase_pri"] = phase[i0]

    idx = mark_phase_range(phase, i0, gap_width)
    conv[idx] = 0

    i1 = int(np.argmin(conv))
    out["sec"] = i1
    out["phase_sec"] = phase[i1]
    idx = mark_phase_range(phase, i1, gap_width)
    conv[idx] = 0

    i2 = np.argmin(conv)
    out["ter"] = i2
    out["phase_ter"] = phase[i2]

    # Gap out 3 transit durations either side of primary and secondary
    # before looking for +ve event
    idx = mark_phase_range(phase, i0, pos_gap_width)
    idx |= mark_phase_range(phase, i1, pos_gap_width)
    conv[idx] = 0

    if np.any(conv):
        i0 = np.argmax(conv)
        out["pos"] = i0
        out["phase_pos"] = phase[i0]
    else:
        out["pos"] = np.nan
        out["phase_pos"] = np.nan

    return out


def mark_phase_range(phase_days, i0, gap_width_days):
    """Mark phase range."""
    if not np.all(np.diff(phase_days) >= 0):
        raise ValueError("Phase not sorted")

    p0 = phase_days[i0]
    period_days = np.max(phase_days)

    idx = p0 - gap_width_days < phase_days
    idx &= phase_days < p0 + gap_width_days

    # Take care of event v close to first phase point
    if p0 < gap_width_days:
        diff = gap_width_days - p0
        idx |= phase_days > period_days - diff

    # Event close to last phase point
    if p0 + gap_width_days > period_days:
        diff = p0 + gap_width_days - period_days
        idx |= phase_days < diff

    return idx


def estimate_scatter(phi_days, flux, phi_pri_days, phi_sec_days,
                     gap_width_hrs):
    """Estimate the point-to-point scatter in the lightcurve after the
    transits have been removed.

    .. todo:: Did Jeff smooth out any residuals in the folded lightcurve
              before computing the scatter?

    Parameters
    ----------
    phi_days, flux : float
        The folded lightcurve
    phi_pri_days, phi_sec_days : float
        Phase of primary and secondary transits, in units of days.
    gap_width_hrs : float
        How much data on either side of primary and secondary
        transits to gap before computing the point-to-point scatter

    Returns
    -------
    rms : float
        The rms point-to-point scatter

    Raises
    ------
    ValueError
        Invalid inputs or calculation failed.

    """
    if len(phi_days) != len(flux):
        raise ValueError('phi_days and flux must be of same length')

    gap_width_days = gap_width_hrs / 24.0

    # Identfiy points near the primary
    idx1 = phi_pri_days - gap_width_days < phi_days
    idx1 &= phi_days < phi_pri_days + gap_width_days

    # ID points near secondary
    idx2 = phi_sec_days - gap_width_days < phi_days
    idx2 &= phi_days < phi_sec_days + gap_width_days

    # Measure rms of all other points
    idx = ~(idx1 | idx2)
    if not np.any(idx):
        raise ValueError('RMS calculation failed')
    rms = np.std(flux[idx])
    return rms


def compute_convolution_for_binned_data(phase, flux, model):
    """Convolve the binned data with the model

    Parameters
    ----------
    phase, flux
        (1d np arrays) Phase folded and binned lightcurve.
        The phase array should be equally spaced.
    model
        (1d np array) Model transit computed at the same
        phase values as the `phase` array

    Returns
    -------
    result
        2d numpy array of convolved data. Columns are phase and flux

    Raises
    ------
    ValueError
        Invalid inputs
    """
    if not np.all(np.isfinite(flux)):
        raise ValueError('flux must contain all finite values')
    if not np.all(np.isfinite(model)):
        raise ValueError('model must contain all finite values')
    if len(phase) != len(flux):
        raise ValueError('phase and flux must be of same length')
    if len(flux) != len(model):
        raise ValueError('model and flux must be of same length')

    # Double up the phase and bflux for shifting
    period = np.max(phase)
    phase = np.concatenate([phase, period + phase])
    flux = np.concatenate([flux, flux])
    # Ensures modshift values are -ve
    conv = np.convolve(flux, -model, mode="valid")

    return conv[:-1]


def compute_phase(time, period, epoch, offset=0):
    """Compute phase."""
    # Make sure the epoch is before the first data point for folding
    delta = epoch - np.min(time)
    if delta > 0:
        epoch -= np.ceil(delta / period) * period

    return np.fmod(time - epoch + offset * period, period)
