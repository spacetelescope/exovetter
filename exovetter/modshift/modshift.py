# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:48:17 2020

Convolve data with model
Id primary, secondary, tertiary, most positive
Gap primary and secondary, measure sigma
Convert conv to significance
Measure sigma pri, sec, ter etc

@author: fergal
"""

from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
import scipy.special as spspec
import numpy as np

import exovetter.modshift.names as names


def compute_modshift_metrics(time, flux, tce, transitModelFunc):
    """Top level function"""
    assert np.all(np.isfinite(time))
    assert np.all(np.isfinite(flux))

    period_days = tce[names.PERIOD]
    epoch_days = tce[names.EPOCH]
    duration_hrs = tce[names.DURATION_HRS]

    overres = 10
    numBins = overres * period_days * 24 / duration_hrs
    numBins = int(numBins)

    offset = 0.25
    data = fold_and_bin_data(time, flux, period_days, epoch_days, offset, numBins)
    bphase = data[:, 0]
    bflux = data[:, 1]
    model = transitModelFunc(bphase, tce, offset)

    conv = compute_convolution_for_binned_data(bphase, bflux, model, offset)
    results = find_indices_of_key_locations(conv, period_days, duration_hrs)

    phi_days = compute_phase(time, period_days, epoch_days, offset)
    sigma = estimate_scatter(
        phi_days, flux, results["pri"], results["sec"], 2 * duration_hrs
    )
    assert sigma > 0
    conv[:, 1] /= sigma

    results.update(compute_event_significances(conv, results))
    key = "false_alarm_threshold"
    results[key] = compute_false_alarm_threshold(period_days, duration_hrs)

    if True:
        plt.clf()
        ax = plt.subplot(211)
        plt.plot(phi_days, 1e3 * flux, "k.")
        plt.plot(bphase, 1e3 * model, "r-", label="Model")
        plt.ylabel("Flux (ppk)")

        mark_events(results)
        plt.legend()

        plt.subplot(212, sharex=ax)
        plt.plot(conv[:, 0], conv[:, 1], "b.")
        mark_events(results)
        mark_false_alarm_threshold(results)

        ymin = 2 * results["sigma_ter"]
        ymax = 2 * results["sigma_pos"]
        plt.ylim(ymin, ymax)
        plt.xlabel("Phase (days)")
        plt.ylabel(r"Significance ($\sigma$)")

    return results


def mark_events(results):
    r = results
    plt.axvline(
        r["pri"], ls="--", color="g", label=r"Primary %.1f$\sigma$" % (r["sigma_pri"])
    )
    plt.axvline(
        r["sec"],
        ls="--",
        color="orange",
        label=r"Secondary %.1f$\sigma$" % (r["sigma_sec"]),
    )
    plt.axvline(
        r["ter"], ls="--", color="m", label=r"Tertiary %.1f$\sigma$" % (r["sigma_ter"])
    )
    plt.axvline(
        r["pos"], ls="--", color="c", label=r"Positive %.1f$\sigma$" % (r["sigma_pos"])
    )


def mark_false_alarm_threshold(results):
    plt.axhline(
        -results["false_alarm_threshold"],
        ls=":",
        color="indigo",
        label="False Alarm Threshold",
    )


def compute_false_alarm_threshold(period_days, duration_hrs):
    duration_days = duration_hrs / 24.0

    fa = spspec.erfcinv(duration_days / period_days)
    fa *= np.sqrt(2)
    return fa


def compute_event_significances(conv, results):
    out = dict()
    for key in "pri sec ter pos".split():
        i0 = np.argmin(np.fabs(conv[:, 0] - results[key]))
        outKey = f"sigma_{key}"
        out[outKey] = conv[i0, 1]

    return out


def find_indices_of_key_locations(conv, period_days, duration_hrs):

    out = dict()
    transit_width = duration_hrs / 24.0
    gap_width = 2 * transit_width
    pos_gap_width = 3 * transit_width

    phase = conv[:, 0].copy()
    ms = conv[:, 1].copy()

    i0 = np.argmin(ms)
    out["pri"] = phase[i0]

    idx = phase[i0] - gap_width < phase
    idx &= phase < phase[i0] + gap_width
    ms[idx] = 0

    i1 = np.argmin(ms)
    out["sec"] = phase[i1]

    idx = phase > phase[i1] - gap_width
    idx &= phase < phase[i1] + gap_width
    ms[idx] = 0

    i2 = np.argmin(ms)
    out["ter"] = phase[i2]

    # Gap out 3 transit durations either side of primary and secondary
    # before looking for +ve event
    idx = phase > phase[i0] - pos_gap_width
    idx &= phase < phase[i0] + pos_gap_width
    ms[idx] = 0

    idx = phase > phase[i1] - pos_gap_width
    idx &= phase < phase[i1] + pos_gap_width
    ms[idx] = 0

    i0 = np.argmax(ms)
    out["pos"] = phase[i0]

    return out


def estimate_scatter(phi_days, flux, phi_pri_days, phi_sec_days, gap_width_hrs):
    """
    phi \\elt [0, period]

    Gap out primary and secondary, measure rms of rest of points
    """
    gap_width_days = gap_width_hrs / 24.0

    # Identfiy points near the primary
    idx1 = phi_pri_days - gap_width_days < phi_days
    idx1 &= phi_days < phi_pri_days + gap_width_days

    # ID points near secondary
    idx2 = phi_sec_days - gap_width_days < phi_days
    idx2 &= phi_days < phi_sec_days + gap_width_days

    # Measure rms of all other points
    idx = ~(idx1 | idx2)
    assert np.any(idx)
    rms = np.std(flux[idx])
    return rms


def compute_convolution_for_binned_data(phase, flux, model, offset_period):
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(model))

    # Double up the phase and bflux for shifting
    period = np.max(phase)
    phase = np.concatenate([phase, period + phase])
    flux = np.concatenate([flux, flux])
    conv = np.convolve(flux, -model)  # Ensures modshift values are -ve

    i0 = int(1 * len(model))
    i1 = i0 + len(model)
    # i1 = len(conv)
    phi = phase[i0:i1]
    conv = conv[i0:i1]
    phi = np.fmod(phi - offset_period * period, period)

    if False:
        plt.clf()
        plt.subplot(311)
        plt.plot(phase, flux, "ko")
        # i0 = 0
        # i1 = int(len(phase)/2)
        # plt.plot(phase[i0:i1], model, 'r-')

        plt.subplot(312)
        # plt.plot(phi, conv, 'b.-')
        plt.plot(conv, "b.-")

        plt.subplot(313)
        plt.plot(phi, conv, "b.-")

        plt.pause(0.1)
        idebug()
    out = np.vstack([phi, conv]).transpose()
    return out


def box_car(time, tce, offset=0):
    """The simplest transit model: A point is either fully in or out of transit

    This is a placeholder for a more realistic model to come
    """

    print(tce)
    period = tce[names.PERIOD]
    epoch = offset * period
    duration_days = tce[names.DURATION_HRS] / 24
    depth_frac = tce[names.DEPTH] * 1e-6

    # Make epoch the start of the transit, not the midpoint
    epoch -= duration_days / 2.0

    mnT = np.min(time)
    mxT = np.max(time)

    e0 = int(np.floor((mnT - epoch) / period))
    e1 = int(np.floor((mxT - epoch) / period))

    flux = 0.0 * time
    for i in range(e0, e1 + 1):
        t0 = period * i + epoch
        t1 = t0 + duration_days
        print(i, t0, t1)

        idx = (t0 <= time) & (time <= t1)
        flux[idx] -= depth_frac

    return flux


def fold_and_bin_data(time, flux, period, epoch, offset, num_bins):
    """Fold data, then bin it up.

    Inputs:

    Returns:

        Notes:
    This isn't everything I want it to be. It assumes that every
    element in y, falls entirely in one bin element, which is not
    necessarily true.

    Secondly, it doesn't weight the bins by the number of elements
    that fall in them.

    I need a better function. Use kplrfits.foldAndBinData() instead
    """
    i = np.arange(num_bins)
    bins = i / float(num_bins) * period  # 0..period in numBin steps

    phase = compute_phase(time, period, epoch, offset)
    srt = np.argsort(phase)
    phase = phase[srt]
    flux = flux[srt]

    cts = np.histogram(phase, bins=bins)[0]
    binnedFlux = np.histogram(phase, bins=bins, weights=flux)[0]
    idx = cts > 0
    binnedFlux = binnedFlux[idx]  # / cts[idx]
    # TODO Trap bins that get no data

    numNonZeroBins = np.sum(idx)
    out = np.zeros((numNonZeroBins, 2))
    out[:, 0] = bins[:-1][idx]
    out[:, 1] = binnedFlux
    return out


def compute_phase_for_tce(time, tce, offset=0.25):
    period = tce[names.PERIOD]
    epoch = tce[names.EPOCH]
    return compute_phase(time, period, epoch, offset)


def compute_phase(time, period, epoch, offset=0.25):
    return np.fmod(time - epoch + offset * period, period)


"""
    #Jeff's way of doing it. I don't think I want to do it that way
  for(i=0;i<ndat;i++)  // Perform ndat pertubations
    {
    tmpsum2 = 0;

    // Before transit, can look up values for compuatation speed increase
    for(j=0;j<startti;j++)
      tmpsum2 += flat[j+i];

    // Compute new values inside transit
    for(j=startti;j<endti;j++)
      tmpsum2 += pow(data[j+i].flux - data[j].model,2);  // Shitfing data, holding model steady. Moving data points backwards, or model forwards, same thing

    // After transit, can look up values for computation speed increase
    for(j=endti;j<ndat;j++)
      tmpsum2 += flat[j+i];

    rms[i] = sqrt(tmpsum2/ndat);  // RMS of the new residuals
    if(rms[i] < rmsmin)
      rmsmin = rms[i];
    }

"""
