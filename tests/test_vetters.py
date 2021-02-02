from numpy.testing import assert_allclose
from astropy.io import ascii
from astropy import units as u
import lightkurve as lk

from exovetter import const as exo_const
from exovetter import vetters
from exovetter.tce import Tce

from astropy.utils.data import get_pkg_data_filename


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


def test_vetters():

    tce = get_wasp18_tce()
    lc = get_wasp18_lightcurve()

    metrics = dict()
    vetter_list = [vetters.Lpp(),
                   vetters.OddEven(),
                   vetters.TransitPhaseCoverage()
                   ]

    for v in vetter_list:
        vetter = v
        _ = vetter.run(tce, lc)
        metrics.update(vetter.__dict__)

    assert_allclose(metrics["norm_lpp"], 7.93119, rtol=1e-3)
    assert_allclose(metrics["tp_cover"], 1.0, rtol=1e-5)
    assert_allclose(metrics["odd_depth"][0], 0.99, rtol=1e-1)


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
    expected = "<test_vetter.ModifiedVetter"
    assert w_str.startswith(expected), w_str
