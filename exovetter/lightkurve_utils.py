"""Functions to deal with the changing lightkurve versions"""

import astropy

def unpack_lk_version(lightcurve, flux_name):
    """Code to get time, flux and time_format from lightcurve object
    independent of v1 or v2 of the package.
    """
    if type(lightcurve.time) is astropy.time.core.Time:
        time = lightcurve.time.value
        flux = getattr(lightcurve, flux_name).value
        time_offset_str = lightcurve.time.format
    else:
        time = lightcurve.time
        flux = getattr(lightcurve, flux_name)
        time_offset_str = lightcurve.time_format

    return time, flux, time_offset_str