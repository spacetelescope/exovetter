import numpy as np
from astropy import units as u
from lightkurve import LightCurve

from exovetter import const as exo_const
from exovetter.tce import Tce
from exovetter.vetters import Sweet
import exovetter.sweet as sweet


# Test produces an error in that lc_read doesn't close the file
# import pytest
# from numpy.testing import assert_allclose
# from lightkurve.search import open as lc_read  # Will be read in 2.x
# from astropy.utils.data import download_file
# @pytest.mark.remote_data
# def test_kplr10417986():
#     """Integration test"""
#     filename = download_file('https://archive.stsci.edu/missions/kepler/lightcurves/0104/010417986/kplr010417986-2010174085026_llc.fits', cache=True)  # noqa

#     period = .0737309 * u.day
#     epoch_mbjd = 55000.027476 * u.day
#     duration = 1 * u.hour
#     target_name = "KPLR 010417986"
#     event_name = "False Positive"

#     epoch_bkjd = epoch_mbjd - exo_const.mbjd + exo_const.bkjd
#     tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=exo_const.bkjd,
#               duration=duration, depth=0 * exo_const.frac_amp,
#               target_name=target_name, event_name=event_name)

#     lcf = lc_read(filename)
#     lc = lcf.PDCSAP_FLUX
#     sweet_vetter = Sweet()
#     res = sweet_vetter.run(tce, lc)
#     amp = res['amp']

#     assert_allclose(amp[0, 0], 637, atol=30)  # Amplitude
#     assert_allclose(amp[0, 2], 106.94, atol=10)  # SNR


def test_sweet_vetter():
    """Tests the interface of the Sweet vetter class, without worrying about the
    correctness of the implementation
    """
    period = 8. * u.day
    epoch_bkjd = 133. * u.day
    duration = 4.0 * u.hour
    target_name = "Dummy Data"
    event_name = "Data used to test the interface"

    tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=exo_const.bkjd,
              duration=duration, depth=0 * exo_const.frac_amp,
              target_name=target_name, event_name=event_name)

    # from lightkurve.lightcurve import LightCurve
    rng = np.random.default_rng(seed=1234)
    time = np.arange(1000)
    flux = 10 + rng.standard_normal(1000)
    lc = LightCurve(time, flux, time_format='bkjd')

    sweet_vetter = Sweet()
    res = sweet_vetter.run(tce, lc)
    sweet_vetter.plot()

    assert isinstance(res, dict)
    assert 'msg' in res.keys()
    assert 'amp' in res.keys()
    amp = res['amp']
    assert amp.ndim == 2
    assert amp.shape == (3, 3)


def test_sine_curve():
    """Implementation test

    Test on a sine curve
    """
    amp = .01
    noise = .001

    transit_period = 100
    sine_period = 50
    res = run_sweet_metric(transit_period, sine_period, amp, noise)
    assert np.isclose(res[0, 0], amp, atol=3 * res[0, 1])

    transit_period = 100
    sine_period = 100
    res = run_sweet_metric(transit_period, sine_period, amp, noise)
    assert np.isclose(res[1, 0], amp, atol=3 * res[1, 1])

    transit_period = 50
    sine_period = 100
    res = run_sweet_metric(transit_period, sine_period, amp, noise)
    assert np.isclose(res[2, 0], amp, atol=3 * res[2, 1]), res


def run_sweet_metric(transit_period, sine_period, amp, noise, plot=False):
    x = np.arange(100, dtype=float)
    y = 1 + amp * np.sin(2 * np.pi * x / sine_period)
    rng = np.random.default_rng(seed=1234)
    y += noise * rng.standard_normal(len(y))

    res = sweet.sweet(x, y, transit_period, 0, 1, plot)
    return res
