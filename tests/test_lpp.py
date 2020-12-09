"""Test LPP functionalities."""
import pytest
from astropy import units as u
from lightkurve import search_lightcurvefile
from numpy.testing import assert_allclose

from exovetter import const
from exovetter import vetters
from exovetter.tce import Tce


# @pytest.mark.skip(reason='Fix the test')
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
              epoch_offset=0 * u.day, snr = 10)

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




#This is more of an example, not a test. I suggest we remove and put
#into an example notebook.
# @pytest.mark.skip(reason='Fix the test')
# def test_run_many_tces():
#     exo = None  # FIXME

#     # Each element of vet_list needs to contain the TCE info
#     # and info to create the ligth curve (filename, or lightkurve call)
#     # In this case we likely will have already created the light curves and
#     # saved them some place. So this is just a list of filenames.
#     vet_list = exo.load_vet_list('filename_tce.csv')

#     Lpp = exo.lpp('path/to/file', lc="DETRENDED")
#     sap_Mod = exo.modshift(model="trap", lc="SAP_FLUX")
#     pdc_Mod = exo.modshift(model="trap", lc="PDC_FLUX")
#     Snr = exo.snr()

#     vetter_list = [Lpp, sap_Mod, pdc_Mod, Snr]

#     # Clearly all this below could also be a method some day.
#     results = list()

#     for vet in vetter_list:
#         # Details of input are hidden here.
#         # Likely the lc information points to a filename (s3 buckeet)
#         # to load data
#         # And put data into the proper format
#         tce, lc = exo.load_tce_and_lightcurve(vet)

#         tce_results = dict()
#         tce_results.update(tce)

#         for v in vetter_list:
#             tce_results.update(v.run(tce, lc))

#         results.append(tce_results)

#     assert len(results) == len(vet_list)


# def run_many_vetters(tce, vetterlist, **kwargs):
#     import lightkurve
#     lc = lightkurve.load_lightcurve_tess(tce, **kwargs)

#     metrics = dict()
#     for v in vetterlist:
#         metrics.update(v.run(tce, lc))

#     return metrics


# @pytest.mark.skip(reason='Fix the test')
# def test_fergal_approach():
#     exo = None  # FIXME
#     load_tce_list = None  # FIXME
#     filename = ''  # FIXME

#     tcelist = load_tce_list(filename)

#     vetterlist = [exo.Lpp('path/to/file')]
#     for tce in tcelist:
#         metrics = run_many_vetters(tce, vetterlist, lctype="sap", detrend=False)  # noqa
