"""Constants and units for exoplanet vetting."""

from astropy import units as u

__all__ = ['bjd', 'bkjd', 'btjd', 'bet', 'ppk', 'ppm', 'frac_amp',
           'string_to_offset']

# TODO: Improve docstrings

# Time offset constants

bjd = 0 * u.day
"""BJD"""

mbjd = bjd - 2_400_000.5 * u.day
"""MJD"""

bkjd = bjd - 2_454_833 * u.day
"""BKJD"""

btjd = bjd - 2_457_000 * u.day
"""BTJD"""

bet = bjd - 2_451_544.5 * u.day
"""Barycentric Emphemeris Time (BET)"""

string_to_offset = dict(
                        bjd=bjd,
                        bkjd=bkjd,
                        btjd=btjd,
                        bet=bet
                        )

# Handy units to express depth

ppk = 1e-3 * u.dimensionless_unscaled
"""PPK"""

ppm = 1e-3 * ppk
"""PPM"""

frac_amp = u.dimensionless_unscaled
"""Frac amp"""

string_to_offset = {'bkjd': bkjd, 'kjd': bkjd,
                    'btjd': btjd,
                    'bjd': bjd, 'mjd': mbjd
                    }
"""Supported Time Offset Keywords"""
