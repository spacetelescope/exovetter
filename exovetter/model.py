"""The ``exovetter.model`` module handles transit models for the
modshift vetter.

Available models:

* Boxcar
* Trapezoid (to be added in the future)

Developer notes
===============

For each model type, create 2 functions:

1. ``create_<modelname>_model_for_tce``: Takes a TCE as input.
2. ``create_<modelname>``: Takes more fundamental times
   (e.g., numpy arrays, floats).

"""
# TODO: See https://github.com/spacetelescope/exovetter/issues/32
# FIXME: Improve docstrings.

import numpy as np
from astropy import units as u

__all__ = ['create_box_model_for_tce', 'create_box_model']


def create_box_model_for_tce(tce, times, time_offset):
    """Create boxcar model for a given TCE.

    Parameters
    ----------
    tce : `~exovetter.tce.Tce`
        Threshold Crossing Event.

    times : `~astropy.units.Quantity`
        Times.

    time_offset : `~astropy.units.Quantity`
        Time offset for :meth:`exovetter.tce.Tce.get_epoch`.

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Flux from boxcar model.

    Raises
    ------
    ValueError
        Invalid input.

    See also
    --------
    create_box_model

    """
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
    """Create boxcar model.

    Parameters
    ----------
    times : array
        Times.

    period : float
        Period.

    epoch : float
        Epoch.

    duration  : float
        Duration.

    depth : `~astropy.units.Quantity`
        Depth.

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Flux from boxcar model, in the same unit as ``depth``.

    See also
    --------
    create_box_model_for_tce

    """
    # Make epoch the start of the transit, not the midpoint
    epoch -= duration * 0.5

    mnT = np.min(times)
    mxT = np.max(times)

    e0 = int(np.floor((mnT - epoch) / period))
    e1 = int(np.floor((mxT - epoch) / period))

    flux = np.zeros_like(times)
    depth_values = depth.to_value()
    for i in range(e0, e1 + 1):
        t0 = period * i + epoch
        t1 = t0 + duration

        idx = (t0 <= times) & (times <= t1)
        flux[idx] -= depth_values

    return flux * depth.unit
