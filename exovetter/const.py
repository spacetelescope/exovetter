"""Constants and units for exoplanet vetting."""

import astropy.units as u

__all__ = ['bjd', 'bkjd', 'btjd', 'bet', 'ppk', 'ppm', 'frac_amp']

# TODO: Improve docstrings

# Time offset constants

bjd = 0 * u.day
"""BJD"""

bkjd = bjd - 2_454_833 * u.day
"""BKJD"""

btjd = bjd - 2_457_000 * u.day
"""BTJD"""

bet = bjd - 2_451_544.5 * u.day
"""Barycentric Emphemeris Time (BET)"""

# Handy units to express depth

ppk = 1e-3 * u.dimensionless_unscaled
"""PPK"""

ppm = 1e-3 * ppk
"""PPM"""

frac_amp = u.dimensionless_unscaled
"""Frac amp"""
