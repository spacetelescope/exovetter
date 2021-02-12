# -*- coding: utf-8 -*-
"""Utilities to deal with lightcurves and lighkurve objects."""

import numpy as np

__all__ = ['set_median_flux_to_zero', 'set_median_flux_to_one']


def set_median_flux_to_zero(flux):
    """Set median flux to zero."""
    if not np.all(np.isfinite(flux)):
        raise ValueError('flux must contain all finite values')

    medflux = np.median(flux)
    if np.isclose(medflux, 0):
        return flux

    flux = flux.copy()
    flux /= medflux
    return flux - 1


def set_median_flux_to_one(flux):
    """Set median flux to one."""
    if not np.all(np.isfinite(flux)):
        raise ValueError('flux must contain all finite values')

    medflux = np.median(flux)
    if np.isclose(medflux, 0):
        return flux + 1

    flux = flux.copy()
    flux /= medflux
