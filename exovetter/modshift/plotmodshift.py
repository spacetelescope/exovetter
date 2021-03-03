# -*- coding: utf-8 -*-
"""Functions used in plotting the results of the modshift calculation."""

import numpy as np

__all__ = ['plot_modshift', 'mark_events', 'mark_false_alarm_threshold']


def plot_modshift(phase, period_days, flux, model, conv, results):
    """Plot modshift results."""
    import matplotlib.pyplot as plt

    srt = np.argsort(phase)
    phase = np.concatenate((phase[srt] - period_days, phase[srt]))
    flux = np.concatenate((flux[srt], flux[srt]))
    model = np.concatenate((model[srt], model[srt]))
    conv = np.concatenate((conv, conv))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(211)
    plt.plot(phase, 1e3 * flux, "k.")
    plt.plot(phase, 1e3 * model, "r-", label="Model")
    plt.ylabel("Flux (ppk)")

    plt.legend()

    plt.subplot(212, sharex=ax)
    x = np.linspace(-1 * period_days, period_days, len(conv)) * np.max(phase)
    plt.plot(x, conv, "b.", label="convolution")
    mark_events(results)
    mark_false_alarm_threshold(results)
    plt.legend()


def mark_events(results):
    """Mark events."""
    import matplotlib.pyplot as plt

    r = results
    plt.axvline(
        r["phase_pri"],
        ls="--",
        color="g",
        label=r"Primary %.1f$\sigma$" % (r["sigma_pri"]),
    )
    plt.axvline(
        r["phase_sec"],
        ls="--",
        color="orange",
        label=r"Secondary %.1f$\sigma$" % (r["sigma_sec"]),
    )
    plt.axvline(
        r["phase_ter"],
        ls="--",
        color="m",
        label=r"Tertiary %.1f$\sigma$" % (r["sigma_ter"]),
    )
    plt.axvline(
        r["phase_pos"],
        ls="--",
        color="c",
        label=r"Positive %.1f$\sigma$" % (r["sigma_pos"]),
    )


def mark_false_alarm_threshold(results):
    """Mark false alarm threshold."""
    import matplotlib.pyplot as plt

    plt.axhline(
        -results["false_alarm_threshold"],
        ls=":",
        color="indigo",
        label="False Alarm Threshold",
    )
    plt.axhline(
        results["false_alarm_threshold"],
        ls=":",
        color="indigo",
    )


def _plot_convolution(phase, flux, bphase, conv):
    """Debugging plot for `compute_convolution_for_binned_data`"""
    import matplotlib.pyplot as plt

    plt.clf()
    plt.subplot(311)
    plt.plot(phase, flux, "ko")

    plt.subplot(312)
    plt.plot(conv, "b.-")

    plt.subplot(313)
    plt.plot(bphase, conv, "b.-")

    plt.pause(0.1)
