# -*- coding: utf-8 -*-

import exovetter.const as const
from exovetter.tce import Tce
import astropy.units as u
import numpy as np
import pytest



def test_required_quantity_missing():
    tce = Tce(period= 25 * u.day)
    assert tce['period'] == 25 * u.day

    with pytest.raises(TypeError):
        tce['depth'] = 1000

    with pytest.raises(TypeError):
        tce = Tce(period = 25)


def test_misc_quantity():
    tce = Tce(kepid = 1234)
    tce['note'] = "This is a comment"

    tce['period'] = 25 * u.day
    tce['kepid']
    tce['note']

def test_epoch():
    tce = Tce(epoch=1000 * u.day, epoch_offset=const.bkjd)

    epoch_btjd = tce.get_epoch(const.btjd).to_value(u.day)
    assert epoch_btjd < 0  #BTJD is after BKJD
    assert np.isclose(epoch_btjd, 2_454_833 - 2_457_000 + 1000)


