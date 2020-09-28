# -*- coding: utf-8 -*-
"""
Functions used in plotting the results of the modshift calculation
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import numpy as np

def plot_modshift(phase, flux, model, conv, results):

    srt = np.argsort(phase)
    phase = phase[srt]
    flux = flux[srt]
    model = model[srt]

    # srt = np.argsort(conv[:,0])
    # conv = conv[srt, :]


    plt.clf()
    ax = plt.subplot(211)
    plt.plot(phase, 1e3 * flux, "k.")
    plt.plot(phase, 1e3 * model, "r-", label="Model")
    plt.ylabel("Flux (ppk)")

    # mark_events(results)
    plt.legend()

    plt.subplot(212, sharex=ax)
    x = np.linspace(0, 1, len(conv)) * np.max(phase)
    plt.plot(x, conv, 'b.')
    # # mark_events(results)
    # # mark_false_alarm_threshold(results)

    # ymin = 2 * results["sigma_ter"]
    # ymax = 2 * results["sigma_pos"]
    # plt.ylim(ymin, ymax)
    # plt.xlabel("Phase (days)")
    # plt.ylabel(r"Significance ($\sigma$)")


def mark_events(results):
    r = results
    plt.axvline(
        r["phase_pri"], ls="--", color="g", label=r"Primary %.1f$\sigma$" % (r["sigma_pri"])
    )
    plt.axvline(
        r["phase_sec"],
        ls="--",
        color="orange",
        label=r"Secondary %.1f$\sigma$" % (r["sigma_sec"]),
    )
    plt.axvline(
        r["phase_ter"], ls="--", color="m", label=r"Tertiary %.1f$\sigma$" % (r["sigma_ter"])
    )
    plt.axvline(
        r["phase_pos"], ls="--", color="c", label=r"Positive %.1f$\sigma$" % (r["sigma_pos"])
    )

def mark_events2(results):
    r = results
    # idebug()
    plt.axvline(
        r["phase_pri"], ls="--", color="g", label=r"Primary %.1f$\sigma$" % (r["sigma_pri"])
    )
    plt.axvline(
        r["phase_sec"],
        ls="--",
        color="orange",
        label=r"Secondary %.1f$\sigma$" % (r["sigma_sec"]),
    )
    plt.axvline(
        r["phase_ter"], ls="--", color="m", label=r"Tertiary %.1f$\sigma$" % (r["sigma_ter"])
    )
    plt.axvline(
        r["phase_pos"], ls="--", color="c", label=r"Positive %.1f$\sigma$" % (r["sigma_pos"])
    )


def mark_false_alarm_threshold(results):
    plt.axhline(
        -results["false_alarm_threshold"],
        ls=":",
        color="indigo",
        label="False Alarm Threshold",
    )


def _plot_convolution(phase, flux, bphase, conv):
    """Debugging plot for `compute_convolution_for_binned_data

    Private functino"""
    plt.clf()
    plt.subplot(311)
    plt.plot(phase, flux, "ko")

    plt.subplot(312)
    # plt.plot(phi, conv, 'b.-')
    plt.plot(conv, "b.-")

    plt.subplot(313)
    plt.plot(bphase, conv, "b.-")

    plt.pause(0.1)
