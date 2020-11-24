import numpy as np
import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose

from exovetter import const
from exovetter.model import create_box_model_for_tce
from exovetter.tce import Tce


@pytest.mark.parametrize(('epoch_val', 'ans_argmin'), [(10, 240), (-10, 2159)])
def test_get_model(epoch_val, ans_argmin):
    period = 100 * u.day
    epoch_bkjd = epoch_val * u.day
    tce = Tce(period=period, epoch=epoch_bkjd, epoch_offset=const.bkjd,
              depth=1 * const.ppk, duration=1 * u.hour)
    times = np.linspace(0 * u.day, period, 2400)
    model = create_box_model_for_tce(tce, times, const.bkjd)

    assert np.sum(model < 0) == 1
    assert np.sum(model > 0) == 0
    assert_quantity_allclose(np.min(model), -1e-3)
    assert np.argmin(model) == ans_argmin
