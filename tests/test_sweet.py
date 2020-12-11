import numpy as np
import pytest
from astropy import units as u
from astropy.utils.data import download_file
from lightkurve import LightCurve
from lightkurve.search import open as lc_read  # Will be read in 2.x
from numpy.testing import assert_allclose

from exovetter import const as exo_const
from exovetter.tce import Tce
from exovetter.vetters import Sweet
import exovetter.sweet as sweet


# @pytest.mark.remote_data
# def test_kepler_1026032():
#     filename = download_file('http://archive.stsci.edu/missions/kepler/lightcurves/0010/001026032/kplr001026032-2009166043257_llc.fits', cache=True)  # noqa

#     period = 8.46043892 * u.day
#     epoch_bkjd = 133.7744903 * u.day
#     duration = 4.73492 * u.hour
#     target_name = "KPLR001026032"
#     event_name = "False Positive"

#     tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=exo_const.bkjd,
#               duration=duration, depth=0 * exo_const.frac_amp,
#               target_name=target_name, event_name=event_name)

#     lcf = lc_read(filename)
#     lc = lcf.PDCSAP_FLUX

#     sweet_vetter = Sweet()
#     sweet_vetter.run(tce, lc)

#     res = sweet_vetter.result
#     assert 'OK' in res['msg']
#     assert_allclose(
#         res['amp'], [[5.13524349e+02, 6.28200510e+02, 8.17452934e-01],
#                      [4.58457276e+02, 6.28225775e+02, 7.29765149e-01],
#                      [2.63828892e+02, 6.28296781e+02, 4.19911258e-01]])


@pytest.mark.remote_data
def test_kplr10417986():
    """Integration test"""
    filename = download_file('https://archive.stsci.edu/missions/kepler/lightcurves/0104/010417986/kplr010417986-2010174085026_llc.fits', cache=True)  # noqa

    period = .0737309 * u.day
    epoch_mbjd = 55000.027476 * u.day
    duration = 1 * u.hour
    target_name = "KPLR 010417986"
    event_name = "False Positive"

    epoch_bkjd = epoch_mbjd - exo_const.mbjd + exo_const.bkjd
    tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=exo_const.bkjd,
              duration=duration, depth=0 * exo_const.frac_amp,
              target_name=target_name, event_name=event_name)

    lcf = lc_read(filename)
    lc = lcf.PDCSAP_FLUX
    sweet_vetter = Sweet()
    res = sweet_vetter.run(tce, lc)
    amp = res['amp']

    assert np.max(amp[:,0]) > 600  #Amplitude is about this
    assert np.max(amp[:,2]) > 100  #Detection SNR should be about this


def test_sweet_vetter():
    """Tests the interface of the Sweet vetter class, without worrying about the
    correctness of the implementation
    """
    period = 8. * u.day
    epoch_bkjd = 133.* u.day
    duration = 4.0 * u.hour
    target_name = "Dummy Data"
    event_name = "Data used to test the interface"

    tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=exo_const.bkjd,
              duration=duration, depth=0 * exo_const.frac_amp,
              target_name=target_name, event_name=event_name)

    # from lightkurve.lightcurve import LightCurve
    time = np.arange(1000)
    flux = 10 + np.random.randn(1000)
    lc = LightCurve(time, flux, time_format='bkjd')

    sweet_vetter = Sweet()
    res = sweet_vetter.run(tce, lc)

    assert isinstance(res, dict)
    assert 'msg' in res.keys()
    assert 'amp' in res.keys()
    amp = res['amp']
    assert amp.ndim == 2
    assert amp.shape == (3,3)

def test_sine_curve():
    """Implementation test

    Test peformance on a sine curve
    """
    amp = .01
    noise = .001

    transit_period = 100
    sine_period = 50
    res = run_sweet_metric(transit_period, sine_period, amp, noise)
    assert np.isclose(res[0,0], amp, atol=3*res[0,1])

    transit_period = 100
    sine_period = 100
    res = run_sweet_metric(transit_period, sine_period, amp, noise)
    assert np.isclose(res[1,0], amp, atol=3*res[1,1])

    transit_period = 50
    sine_period = 100
    res = run_sweet_metric(transit_period, sine_period, amp, noise)
    assert np.isclose(res[2,0], amp, atol=3*res[2,1]), res


def run_sweet_metric(transit_period, sine_period, amp, noise, plot=False):
    x = np.arange(100, dtype=float)
    y = 1 + amp * np.sin(2 * np.pi * x / sine_period)
    y += noise * np.random.randn(len(y))


    res = sweet.sweet(x, y, transit_period, 0, 1, plot)
    return res