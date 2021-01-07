# -*- coding: utf-8 -*-
"""
Utilities to deal with lightcurves and lighkurves
"""

import numpy as np


def set_median_flux_to_zero(flux):
    assert np.all(np.isfinite(flux))

    medflux = np.median(flux)
    if np.isclose(medflux, 0):
        return flux

    flux = flux.copy()
    flux /= medflux
    return flux - 1


def set_median_flux_to_one(flux):
    assert np.all(np.isfinite(flux))

    medflux = np.median(flux)
    if np.isclose(medflux, 0):
        return flux + 1

    flux = flux.copy()
    flux /= medflux
