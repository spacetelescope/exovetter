# -*- coding: utf-8 -*-

import exovetter.const as const
import astropy.units as u
import numpy as np



from exovetter.tce import Tce
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


def test_get_model():
    period_days = 100
    epoch_bkjd = 10
    tce = Tce(period = period_days * u.day,
              epoch = epoch_bkjd * u.day,
              epoch_offset=const.bkjd)
    tce['depth'] = 1 * const.ppk
    tce['duration'] = 1 * u.hour

    times = np.linspace(0, period_days, period_days*24)
    model = tce.get_model(times * u.day, const.bkjd)
    model = model.to_value()

    assert np.sum(model < 0) == 1
    assert np.sum(model > 0) == 0
    assert np.isclose(np.min(model), -1e-3), np.min(model)
    assert np.argmin(model) == 10*24, np.argmin(model)


def test_get_model_negative_epoch():
    period_days = 100
    epoch_bkjd = -10
    tce = Tce(period = period_days * u.day,
              epoch = epoch_bkjd * u.day,
              epoch_offset=const.bkjd)
    tce['depth'] = 1 * const.ppk
    tce['duration'] = 1 * u.hour

    times = np.linspace(0, period_days, period_days*24)
    model = tce.get_model(times * u.day, const.bkjd)
    model = model.to_value()

    assert np.sum(model < 0) == 1
    assert np.sum(model > 0) == 0
    assert np.isclose(np.min(model), -1e-3), np.min(model)
    assert np.argmin(model) == 90*24 - 1, np.argmin(model)


def test_get_model_different_offset():
    period_days = 100
    epoch_bkjd = 10
    tce = Tce(period = period_days * u.day,
              epoch = epoch_bkjd * u.day,
              epoch_offset=const.bkjd)
    tce['depth'] = 1 * const.ppk
    tce['duration'] = 1 * u.hour

    times = np.linspace(0, period_days, period_days*24)
    model = tce.get_model(times * u.day, const.btjd)

    t0_bjd = 2_454_833 + epoch_bkjd
    t0_btjd = t0_bjd - 2_457_000
    # idebug()
    print("t0_btjd =  %g" %(t0_btjd))
    t0_btjd = np.remainder(t0_btjd, period_days)
    print("phase =  %g" %(t0_btjd))

    i0 = np.argmin(model)
    assert np.isclose(times[i0], t0_btjd, atol=1)
