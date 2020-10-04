# -*- coding: utf-8 -*-

import exovetter.const as const
import astropy.units as u
import numpy as np



class Tce(dict):
    required_quantities = set("period epoch epoch_offset duration depth".split())
    def __init__(self, **kwargs):
        dict.__init__(self)
        for k in kwargs:
            self.__setitem__(k, kwargs[k])

    def __setitem__(self, key, value):
        if key in self.required_quantities:
            if not isinstance(value, u.Quantity):
                raise TypeError("Special parameter %s must be an astropy quantity" %(key))
        self.setdefault(key, value)

    def get_epoch(self, offset):
        """Returns an astropy.unit.Quantity"""
        return self['epoch'] - self['epoch_offset'] + offset

    def validate(self):
        is_ok = True

        for q in self.required_quantities:
            if q not in self:
                print("Required quantitiy %s is missing" %(q))
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
        depth = self['depth']  #Keep units attached

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
    required_quantities = set("period epoch duration depth ingress_time".split())

    def get_model(self, times, epoch_offset):
        raise NotImplementedError


