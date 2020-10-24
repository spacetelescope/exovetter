# -*- coding: utf-8 -*-
import exovetter.const as const
from exovetter.tce import Tce
import astropy.units as u
import exovetter.model
import numpy as np


def test_get_model():
    period_days = 100
    epoch_bkjd = 10
    tce = Tce(
        period=period_days * u.day, epoch=epoch_bkjd * u.day,
        epoch_offset=const.bkjd
    )
    tce["depth"] = 1 * const.ppk
    tce["duration"] = 1 * u.hour

    times = np.linspace(0, period_days, period_days * 24)
    model = exovetter.model.create_box_model_for_tce(tce, times * u.day,
                                                     const.bkjd)
    model = model.to_value()

    assert np.sum(model < 0) == 1
    assert np.sum(model > 0) == 0
    assert np.isclose(np.min(model), -1e-3), np.min(model)
    assert np.argmin(model) == 10 * 24, np.argmin(model)


def test_get_model_negative_epoch():
    period_days = 100
    epoch_bkjd = -10
    tce = Tce(
        period=period_days * u.day, epoch=epoch_bkjd * u.day,
        epoch_offset=const.bkjd
    )
    tce["depth"] = 1 * const.ppk
    tce["duration"] = 1 * u.hour

    times = np.linspace(0, period_days, period_days * 24)
    model = exovetter.model.create_box_model_for_tce(tce, times * u.day,
                                                     const.bkjd)
    model = model.to_value()

    assert np.sum(model < 0) == 1
    assert np.sum(model > 0) == 0
    assert np.isclose(np.min(model), -1e-3), np.min(model)
    assert np.argmin(model) == 90 * 24 - 1, np.argmin(model)
