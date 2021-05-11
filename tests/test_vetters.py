from numpy.testing import assert_allclose
import numpy as np

from exovetter.centroid import centroid as cent
from exovetter import const as exo_const
from exovetter import vetters
from exovetter.tce import Tce

from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.io import ascii
import lightkurve as lk


def get_wasp18_tce():

    tce = Tce(
        period=0.94124 * u.day,
        epoch=58374.669883 * u.day,
        epoch_offset=-2400000.5 * u.day,
        depth=0.00990112 * exo_const.frac_amp,
        duration=0.08932 * u.day,
        event_name="WASP-18 b",
        target_name="WASP-18",
        snr=50,
    )

    return tce


def get_wasp18_lightcurve():

    lc_file = get_pkg_data_filename("data/wasp18b_flat_lightcurve.csv")

    lc_table = ascii.read(lc_file, data_start=1)

    lc = lk.LightCurve(
        time=lc_table["col2"],
        flux=lc_table["col3"],
        flux_err=lc_table["col4"],
        time_format="btjd",
    )

    return lc

def test_name():
    v = vetters.OddEven()
    assert v.name() == "OddEven", v.name()

def test_vetters():

    tce = get_wasp18_tce()
    lc = get_wasp18_lightcurve()

    results = dict()
    vetter_list = [vetters.Lpp(),
                   vetters.OddEven(),
                   vetters.TransitPhaseCoverage()
                   ]

    results = dict()
    for v in vetter_list:
        res = v.run(tce, lc)
        results[v.name()] = res

    assert set("Lpp OddEven TransitPhaseCoverage".split()) == set(results.keys())
    for k in results:
        assert isinstance(results[k], dict)

def test_cent_vetter():

    tce = get_wasp18_tce()
    lc = get_wasp18_lightcurve()

    time = lc.time.value
    time_offset_str = lc.time.format
    px_size = (len(time), 7, 9)

    cube = 2000.0 * np.ones(px_size) + \
        np.random.normal(loc=0, scale=100, size=px_size)
    cube[:, 3, 4] = 3901.1 + np.zeros(len(time))
    cube[:, 4, 4] = 3202.5 + np.zeros(len(time))

    period_days = tce["period"].to_value(u.day)
    time_offset_q = getattr(exo_const, time_offset_str)
    epoch = tce.get_epoch(time_offset_q).to_value(u.day)
    duration_days = tce["duration"].to_value(u.day)

    centroids, figs = cent.compute_diff_image_centroids(
        time, cube, period_days, epoch, duration_days, plot=False
    )
    offset, signif, fig = cent.measure_centroid_shift(centroids, plot=False)

    assert len(centroids) == 21
    assert offset < 9


class DefaultVetter(vetters.BaseVetter):
    def run(self, tce, lightcurve):
        pass


class ModifiedVetter(vetters.BaseVetter):
    def __init__(self, **kwargs):
        pass

    def run(self, tce, lightcurve):
        pass


def test_string_dunder():
    """Test that the vetter's string method behaves as expected.

    Note: We may choose to improve the string representation at some point
    """

    v = DefaultVetter()
    v_str = str(v)

    # No metrics gets returned as an empty dictionary
    assert v_str == "{}", v_str

    # A metrics dictionary gets returned as a pprinted string
    v.metrics = dict(key="value")
    v_str = str(v)
    assert v_str == "{'key': 'value'}", v_str

    w = ModifiedVetter()
    w_str = str(w)
    expected = "<test_vetters.ModifiedVetter"
    assert w_str.startswith(expected), w_str
