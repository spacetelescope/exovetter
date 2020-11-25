import pytest
from astropy import units as u
from astropy.utils.data import download_file
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
