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


@pytest.mark.remote_data
def test_kepler_1026032():
    filename = download_file('http://archive.stsci.edu/missions/kepler/lightcurves/0010/001026032/kplr001026032-2009166043257_llc.fits', cache=True)  # noqa

    period = 8.46043892 * u.day
    epoch_bkjd = 133.7744903 * u.day
    duration = 4.73492 * u.hour
    target_name = "KPLR001026032"
    event_name = "False Positive"

    tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=exo_const.bkjd,
              duration=duration, depth=0 * exo_const.frac_amp,
              target_name=target_name, event_name=event_name)

    lcf = lc_read(filename)
    lc = lcf.PDCSAP_FLUX

    sweet_vetter = Sweet()
    sweet_vetter.run(tce, lc)

    res = sweet_vetter.result
    assert 'OK' in res['msg']
    assert_allclose(
        res['amp'], [[5.13524349e+02, 6.28200510e+02, 8.17452934e-01],
                     [4.58457276e+02, 6.28225775e+02, 7.29765149e-01],
                     [2.63828892e+02, 6.28296781e+02, 4.19911258e-01]])


@pytest.mark.parametrize('sine_type', ('single', 'double'))
def test_sine_wave(sine_type):
    period = 0.61 * u.day
    epoch = 0.2 * u.day
    duration = 0.1 * u.day
    time = np.arange(0, 3, 2 / 24) * u.day

    if sine_type == 'single':
        x = 2 * np.pi * (time + epoch) / period
        amp = [[2.55025899e-04, 1.73406992e-04, 1.47067829e+00],
               [9.63733192e-04, 2.95202543e-05, 3.26465071e+01],
               [6.67236445e-05, 1.79074673e-04, 3.72602354e-01]]
        warn_msg = 'WARN: SWEET test finds signal at the transit period'
    else:  # double
        x = 2 * np.pi * (time + epoch) / period * 2
        amp = [[1.00007544e-03, 1.58445948e-06, 6.31177669e+02],
               [3.95471263e-04, 1.73162859e-04, 2.28381112e+00],
               [1.37722120e-04, 1.85663338e-04, 7.41784145e-01]]
        warn_msg = 'WARN: SWEET test finds signal at HALF transit period'

    flux = u.Quantity(0.001 * np.sin(x.to_value()))
    tce = Tce(period=period, epoch=epoch, epoch_offset=exo_const.btjd,
              duration=duration, depth=0 * exo_const.frac_amp)
    lc = LightCurve(time, flux, time_format='btjd', label='Sin-wave')

    sweet_vetter = Sweet()
    sweet_vetter.run(tce, lc)
    res = sweet_vetter.result
    assert res['msg'] == warn_msg
    assert_allclose(res['amp'], amp, rtol=2e-7)
