# -*- coding: utf-8 -*-
"""Module to handle Threshold Crossing Event (TCE).

This module constains a `~exovetter.tce.Tce` class, which stores the measured
properties (orbital period, transit depth, etc.) of a proposed transit.

To create model transits from a `~exovetter.tce.Tce`, see the
`exovetter.model` module. For example, you can obtain flux from a boxcar
model using :func:`~exovetter.model.create_box_model_for_tce`.

Examples
--------

Define a TCE in BKJD:

>>> from astropy import units as u
>>> from exovetter import const as exo_const
>>> from exovetter.model import create_box_model_for_tce
>>> from exovetter.tce import Tce
>>> period = 5.3 * u.day
>>> epoch = 133.4 * u.day
>>> depth = 1 * exo_const.ppm
>>> duration = 24 * u.hr
>>> my_tce = Tce(period=period, epoch=epoch, epoch_offset=exo_const.bkjd,
...              depth=depth, duration=duration, comment='test')
>>> my_tce
{'period': <Quantity 5.3 d>,
 'epoch': <Quantity 133.4 d>,
 'epoch_offset': <Quantity -2454833. d>,
 'depth': <Quantity 1.e-06>,
 'duration': <Quantity 24. h>,
 'comment': 'test'}

Retrieve the epoch of the transit in BJD:

>>> epoch_bjd = my_tce.get_epoch(exo_const.bjd)
>>> epoch_bjd
<Quantity 2454966.4 d>

Calculate flux from boxcar model:

>>> times = [134, 135, 136] * u.d
>>> create_box_model_for_tce(my_tce, times, epoch_bjd)
<Quantity [ 0.e+00, -1.e-06,  0.e+00]>

"""
import json
import astropy.units as u
import exovetter.const as exo_const

__all__ = ['Tce']


class Tce(dict):
    """Class to handle Threshold Crossing Event (TCE).

    It inherits from :py:obj:`dict` and defines a list of reserved
    keywords in ``required_quantities``, which must contain
    `~astropy.units.Quantity`. Other keys in the dictionary have
    no restrictions.

    By requiring that certain measured properties have units attached,
    it reduces the risk of modeling a transit with data
    in the wrong units; e.g., entering the transit duration in
    hours but treating the value as if it is the duration in days.
    It also allows depth given in parts per million to be used correctly
    in places that expect depth to be in fractional amplitude.

    This class also handles the problem with transit times given in
    different zeropoints. Most transit data has times corrected to
    the barycentre of the solar system and expressed in the unit of
    days since some zeropoint. However, there is no agreement
    on the zeropoint. Some data is given in Julian Date, while
    other missions choose mission specific zeropoints; e.g.,
    ``t=0`` for Kepler data corresponds to a Barycentric Julian Date
    (BJD) of 2454833. This problem is addressed here by requiring
    epoch and period data to be `~astropy.units.Quantity` and by
    providing a :meth:`~exovetter.tce.Tce.get_epoch` method.

    Attributes
    ----------
    required_quantities : set
        Keys in which their values must be `~astropy.units.Quantity`.

    Raises
    ------
    KeyError
        Required quantities missing, causing
        :meth:`~exovetter.tce.Tce.validate` to fail.

    """
    required_quantities = {'depth', 'duration', 'epoch', 'epoch_offset',
                           'period'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def __setitem__(self, key, value):
        if (key in self.required_quantities and
                not isinstance(value, u.Quantity)):
            raise TypeError(f"Special param {key} must be an astropy Quantity")
        dict.__setitem__(self, key, value)

    def get_epoch(self, offset=None):
        """Get the epoch of the transit in the desired BJD-based time system.

        The time of first transit of your TCE is stored as a date offset
        from BJD=0 by an amount given by ``offset`` (e.g., BKJD, TJD).
        This method converts the time of the first transit to a time offset
        from the epoch desired.

        Parameters
        ----------
        offset : `~astropy.units.Quantity`
            The epoch offset desired.
            Some pre-defined offsets are available in `exovetter.const`.
            For example, the time of first transit in `~exovetter.const.bkjd`.

        Returns
        -------
        epoch : `~astropy.units.Quantity`
            Epoch of the transit in the desired BJD-based time system.

        Raises
        ------
        KeyError
            Calculation failed due to missing keys.

        """
        if 'epoch' not in self or 'epoch_offset' not in self:
            raise KeyError('epoch and epoch_offset must be defined first')
        epoch = self["epoch"] - self["epoch_offset"]
        if offset is not None:
            epoch = epoch + offset
        return epoch

    def validate(self):
        """Check that required quantities are present.

        Returns
        -------
        status : bool
            `True` if validation passes.

        Raises
        ------
        KeyError
            Some required quantities are missing.

        TyperError
            Some required quantities do not have units attached.

        """
        missing_keys = self.required_quantities - set(self.keys())
        if len(missing_keys) != 0:
            raise KeyError(f'Missing required quantities: {missing_keys}')
        for key in self.required_quantities:
            if not isinstance(self[key], u.Quantity):
                raise TypeError(f"Special param {key} must be an astropy "
                                "Quantity")
        return True

    def to_json(self, filename):
        """Write a json file with standard file name

        Parameters
        ----------
        filename : string
        Filename to write the json string.

        Returns
        -------
        tce_json : json formatted string

        """
        tmp = {}
        for key in self.keys():
            if isinstance(self[key], u.Quantity):
                tmp[key] = self[key].value
                unit_key = f'{key}_unit'
                tmp[unit_key] = str(self[key].unit)
            else:
                tmp[key] = self[key]

        tce_json = json.dumps(tmp)

        if filename is not None:
            with open(filename, 'w') as fobj:
                fobj.write(tce_json)

        return tce_json

    @classmethod
    def from_json(cls, filename):
        """Read a json file and populate a TCE object

        Parameters
        ----------
        filename : string
            Filename of json file containing tce informamtion

        Returns
        -------
        None.

        """
        with open(filename, 'r') as fobj:
            jobj = json.load(fobj)

        tmp = {}

        for key in jobj.keys():
            if key[-5:] == '_unit':
                pass
            else:
                v = jobj[key]
                if key + "_unit" in jobj.keys():
                    unit_str = jobj[key + "_unit"]
                    if unit_str == "":
                        q = exo_const.frac_amp
                    else:
                        q = u.__dict__[unit_str]
                    tmp[key] = v * q
                else:
                    tmp[key] = v

        tce = cls(**tmp)

        return tce
