import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from exovetter import odd_even as oe
from exovetter import tce


def test_odd_even():
    """Simple test of folding with no odd even and no error."""
    t = np.arange(0, 10)
    f = np.ones(len(t))
    f[1::2] = 0.5
    period = 2
    sigma, odd, even = oe.calc_odd_even(t, f, period, epoch=1, duration=1)

    assert_allclose(odd[0], 0.5)
    assert_allclose(even[0], 0.5)
    assert np.isnan(sigma)


def test_odd_even2():
    n = 100
    rng = np.random.default_rng(seed=1234)
    a = rng.standard_normal(n) + 2
    b = rng.standard_normal(n) + 10

    odd = (np.mean(a), np.std(a))
    even = (np.mean(b), np.std(b))

    diff, error, sigma = oe.calc_diff_significance(odd, even)

    assert_allclose(diff, 8, atol=1)
    assert_allclose(error, np.sqrt(2), rtol=0.1)
    assert_allclose(sigma, 5.6, atol=1)


@pytest.mark.skip('FIXME: Enable after the new TCE class is settled upon')
@pytest.mark.parametrize(('noise', 'ans_sigma'), [(0.009, 5), (0.1, 3)])
def test_odd_even_tce(noise, ans_sigma):
    times = np.arange(0, 400, .033)
    period = 100
    duration = 0.5
    epoch = 0.0

    atce = tce.TCE(period * u.day, tzero=epoch, duration=duration * u.day,
                   depth=0.12, target_name='sample', event_name="sample b")
    flux1 = atce.get_boxmodel(times)

    atce = tce.TCE(period * u.day, tzero=epoch + 0.5 * period,
                   duration=duration * u.day, depth=0.5, target_name='sample',
                   event_name="sample b")
    flux2 = atce.get_boxmodel(times)

    rng = np.random.default_rng(seed=1234)
    flux = (flux1 + flux2) * 0.5 + rng.standard_normal(len(flux1)) * noise
    sigma, odd, even = oe.calc_odd_even(
        times, flux, period * 0.5, epoch, duration)

    if noise < 0.01:
        assert sigma > ans_sigma
    else:
        assert sigma < ans_sigma
    assert odd[1] != 1
    assert even[1] != 1
