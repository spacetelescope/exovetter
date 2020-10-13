# -*- coding: utf-8 -*-
"""
Repository of transit models.

Currently, only the box car model is implemented, need a
trapezoid model here too.

For each model type create 2 functions, one that takes a TCE as input,
and one that takes more fundamental times (numpy arrays, floats, etc.)
"""

import astropy.units as u
import numpy as np


def create_box_model_for_tce(tce, times, time_offset):
    if not tce.validate():
        raise ValueError("Required quantities missing from TCE")

    if not isinstance(times, u.Quantity):
        raise ValueError("times is not a Quantity. Please supply units")

    if not isinstance(time_offset, u.Quantity):
        raise ValueError("time_offset is not a Quantity. Please supply units")

    unit = times.unit
    times = times.to_value()

    period = tce["period"].to_value(unit)
    epoch = tce.get_epoch(time_offset).to_value(unit)
    duration = tce["duration"].to_value(unit)
    depth = tce["depth"]  # Keep units attached

    return create_box_model(times, period, epoch, duration, depth)


def create_box_model(times, period, epoch, duration, depth):
    # Make epoch the start of the transit, not the midpoint
    epoch -= duration / 2.0

    mnT = np.min(times)
    mxT = np.max(times)

    e0 = int(np.floor((mnT - epoch) / period))
    e1 = int(np.floor((mxT - epoch) / period))

    flux = 0.0 * times
    for i in range(e0, e1 + 1):
        t0 = period * i + epoch
        t1 = t0 + duration
        # print(i, t0, t1)

        idx = (t0 <= times) & (times <= t1)
        flux[idx] -= depth.to_value()

    return flux * depth.unit
