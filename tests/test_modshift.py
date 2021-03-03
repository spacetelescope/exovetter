# -*- coding: utf-8 -*-

import exovetter.modshift.modshift as modshift
import numpy as np


def test_modshift():
    np.random.seed(1234)
    x = np.arange(100) + 0.25
    y = np.random.randn(100)
    y[40:50] -= 5  # Primary
    y[80:90] -= 3  # secondary

    model = np.zeros(len(x))
    model[40:48] = -5

    period_days = len(x)
    epoch = 44
    duration_hrs = 10 * 24

    res, conv = modshift.compute_modshift_metrics(
        x, y, model, period_days, epoch, duration_hrs
    )

    assert np.isclose(res["pri"], 0, atol=1) or \
        np.isclose(res["pri"], 99, atol=1), res
    assert np.isclose(res["sec"], 84 - epoch, atol=2), res
    return res


def test_single_epoch_sigma():
    np.random.seed(1234)
    x = np.arange(200) + 0.25
    y = np.random.randn(200)
    y[40] -= 5  # Primary
    y[160] -= 3
    model = np.zeros(len(x))

    model[40] = -5
    period_days = len(x)
    epoch = 40
    duration_hrs = 1 * 24

    res, conv = modshift.compute_modshift_metrics(
        x, y, model, period_days, epoch, duration_hrs
    )
    assert np.isclose(res["sigma_pri"], -10, atol=2), res["sigma_pri"]


def test_multi_epoch_sigma():
    # np.random.seed(1234)
    x = np.arange(200) + 0.25
    y = np.random.randn(200)
    y[40] -= 5  # Primary
    y[90] -= 5  # Primary
    y[140] -= 5  # Primary
    y[190] -= 5  # Primary
    # y[80] -= 3 #secondary

    model = np.zeros(len(x))

    model[40] = -5
    model[90] = -5
    model[140] = -5
    model[190] = -5
    period_days = len(x) / 4
    epoch = 40
    duration_hrs = 1 * 24

    res, conv = modshift.compute_modshift_metrics(
        x, y, model, period_days, epoch, duration_hrs
    )
    assert np.isclose(res["sigma_pri"], -40, atol=4), res["sigma_pri"]


def test_fold_and_bin_data():
    x = np.arange(100) + 0.25
    y = np.random.randn(100)

    data = modshift.fold_and_bin_data(x, y, 100, 0, 500)
    assert len(data) == len(x), len(data)

    data = modshift.fold_and_bin_data(x, y, 100, 44, 500)
    assert len(data) == len(x), len(data)

    # idebug()
    data = modshift.fold_and_bin_data(x, y, 100, 44, 100)
    assert len(data) == len(x), len(data)


def test_convolve():
    size = 24
    phase = np.arange(size)
    flux = np.zeros(size)
    flux[0] = -1
    flux[-1] = -1
    flux[8:10] = -0.5

    model = np.zeros(size)
    model[0] = -1
    model[-1] = -1

    conv = modshift.compute_convolution_for_binned_data(phase, flux, model)
    assert len(conv) == len(phase), len(conv)

    print(conv)
    assert np.isclose(conv[0], -2), conv[0]
    assert np.isclose(conv[9], -1, atol=1e-3), conv[9]
