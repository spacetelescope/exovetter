# -*- coding: utf-8 -*-

"""
A TCE class stores the measured properties of a proposed transit.
Those properties include orbital period, transit depth etc.

This class tries to reduce the risk of modeling a transit
with data in the wrong unit, for example, entering the transit duration in
hours, then treating the value as if it is the duration in days.

A Tce class is a dictionary with a list of reserved keywords. The
values of these keys must be astropy.units.Quantities objects. Other
keys in the dictionary have no restrictions on their values. By ensuring
that certain measured quantities are have included units, we can ensure
that, for example, a TCE created with depth measured in parts per million,
is used correctly in code that expects depths to be measured in fractional
amplitude.

Transit times represent a wrinkle in this model. Most transit data has
times corrected to the barycentre of the solar system, and expressed in
units of days since some zero point. However, there is no agreement
on the zero point. Some data is given in Julian Date, other missions choose
mission specific zero points. For example, t=0 for Kepler data corresponds
to a barycentric julian date of 2,454,833.0. Some common offsets are stored
in const.py

The Tce class addresses these zero points by making `epoch_offset` a
reserved keyword. When creating a Tce, you must specify the period and
epoch of the transit (typically with units of days), but also the time
of the zero point of the time system.

Example
----------
::

    period_days = 5.3
    epoch_days = 133.4
    tce = Tce(period=periods_days * u.day,
              epoch=epoch_days * u.day,
              epoch_offset=const.bkjd)


You can retrive the epoch of the transit with the `get_epoch()` method.::

    # Even though the Tce is created with transit time in BKJD, getting the
    # Julian date of the transit is easy:
    epoch_bjd = tce.get_epoch(const.bjd)

The Tce class also lets you compute a model transit based on the input
parameters. Again, you have specify what zero point your required times
are at::

    time, flux = load_kepler_data(...)

    assert isinstance(time, np.ndarray)
    assert dtype(time) == np.float

    # Get the model in the BKJD time system
    model = tce.get_model(time * u.day, const.bkjd)

    assert isinstance(model, u.Quantity)


The default transit model created by a Tce is a box model where a point
is either in, or out, of transit. Subclass the Tce class to create
more sophisticated models, such as trapezoids, or limb-darkened models.
"""

import astropy.units as u
import numpy as np


class Tce(dict):
    required_quantities = "period epoch epoch_offset duration depth".split()
    required_quantities = set(required_quantities)

    def __init__(self, **kwargs):
        dict.__init__(self)
        for k in kwargs:
            self.__setitem__(k, kwargs[k])

    def __setitem__(self, key, value):
        if key in self.required_quantities:
            if not isinstance(value, u.Quantity):
                msg = "Special param %s must be an astropy quantity" % (key)
                raise TypeError(msg)
        self.setdefault(key, value)

    def get_epoch(self, offset):
        """Returns an astropy.unit.Quantity"""
        return self['epoch'] - self['epoch_offset'] + offset

    def validate(self):
        is_ok = True

        for q in self.required_quantities:
            if q not in self:
                print("Required quantitiy %s is missing" % (q))
                is_ok = False
        return is_ok

    def get_model(self, times, epoch_offset):
        if not self.validate():
            raise ValueError("Required quantities missing from TCE")

        if not isinstance(times, u.Quantity):
            raise ValueError("times is not a Quantity. Please supply units")

        unit = times.unit
        times = times.to_value()

        period = self['period'].to_value(unit)
        epoch = self.get_epoch(epoch_offset).to_value(unit)
        duration = self['duration'].to_value(unit)
        depth = self['depth']  # Keep units attached

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


class BoxCarTce(Tce):
    pass


class TrapezoidTce(Tce):
    required_quantities = "period epoch duration depth ingress_time".split()
    required_quantities = set(required_quantities)

    def get_model(self, times, epoch_offset):
        raise NotImplementedError
