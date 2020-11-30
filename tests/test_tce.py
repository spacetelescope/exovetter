import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose

from exovetter import const
from exovetter.tce import Tce


def test_required_quantity_missing():
    with pytest.raises(KeyError, match='Missing required quantities'):
        Tce(period=25 * u.day)

    with pytest.raises(TypeError, match='Special param period must be an '
                       'astropy Quantity'):
        Tce(period=25, epoch=0 * u.day, epoch_offset=0 * u.day,
            duration=1 * u.hr, depth=1 * const.ppk)

    tce = Tce(period=25 * u.day, epoch=0 * u.day, epoch_offset=0 * u.day,
              duration=1 * u.hr, depth=1 * const.ppk)

    with pytest.raises(TypeError, match='Special param depth must be an '
                       'astropy Quantity'):
        tce["depth"] = 1000


def test_misc_quantity():
    tce = Tce(kepid=1234, period=25 * u.day, epoch=0 * u.day,
              epoch_offset=0 * u.day, duration=1 * u.hr, depth=1 * const.ppk)
    tce["note"] = "This is a comment"

    assert_quantity_allclose(tce["period"], 25 * u.day)
    assert tce["kepid"] == 1234
    assert tce["note"] == "This is a comment"


def test_epoch():
    tce = Tce(period=25 * u.day, epoch=1000 * u.day, epoch_offset=const.bkjd,
              duration=1 * u.hr, depth=1 * const.ppk)

    # BTJD is after BKJD
    epoch_btjd = tce.get_epoch(const.btjd)
    assert_quantity_allclose(
        epoch_btjd, (2_454_833 - 2_457_000 + 1000) * u.day)
