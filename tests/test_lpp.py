"""Test LPP functionalities."""
import pytest
from astropy import units as u
from lightkurve import search_lightcurvefile
from numpy.testing import assert_allclose

from exovetter import const
from exovetter import vetters
from exovetter.tce import Tce


@pytest.mark.remote_data
def test_one_lpp():
    """"Use case is to get values for one TCE."""

    period = 3.5224991 * u.day
    tzero = (54953.6193 + 2400000.5 - 2454833.0) * u.day
    duration = 3.1906 * u.hour
    depth = 0.009537 * const.frac_amp
    target_name = "Kepler-8"
    event_name = "Kepler-8 b"

    tce = Tce(period=period, epoch=tzero, duration=duration,
              target_name=target_name, depth=depth, event_name=event_name,
              epoch_offset=0 * u.day, snr=10)

    # Specify the lightcurve to vet
    mission = "Kepler"
    q = 4

    # Generic function that runs lightkurve and returns a lightkurve object
    lcf = search_lightcurvefile(
        target_name, quarter=q, mission=mission).download()
    lc = lcf.SAP_FLUX.remove_nans().remove_outliers()
    flat = lc.flatten(window_length=81)
    flat.flux = flat.flux - 1.0

    # Use default .mat file from SourceForge
    lpp = vetters.Lpp(lc_name="flux", map_filename=None)

    _ = lpp.run(tce, flat)

    # Accepted value if data doesn't change
    assert_allclose(lpp.norm_lpp, 0.17, atol=0.09)
