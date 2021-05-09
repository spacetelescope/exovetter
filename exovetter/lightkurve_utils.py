"""Functions to handle compatibility with lightkurve."""

import astropy

__all__ = ['unpack_lk_version']


def unpack_lk_version(lightcurve, flux_name):
    """Code to get time, flux and time_format from lightcurve object
    independent of v1 or v2 of the package.
    """
    if isinstance(lightcurve.time, astropy.time.core.Time):
        time = lightcurve.time.value
        flux = getattr(lightcurve, flux_name).value
        time_offset_str = lightcurve.time.format
    else:
        time = lightcurve.time
        flux = getattr(lightcurve, flux_name)
        time_offset_str = lightcurve.time_format

    return time, flux, time_offset_str


def unpack_tpf(tpf, name):
    """


    Parameters
    ----------
    tpf : lightkurve object
        lightkurve target pixel file object
    name : string
        name of column with pixels

    Returns
    -------
    cube : numpy array
        pixels as a data cube
    time : numpy array
        times

    """
    time = tpf.time.value
    cube = getattr(tpf, name).value
    time_offset_str = tpf.time.format

    return time, cube, time_offset_str
