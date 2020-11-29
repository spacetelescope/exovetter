#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from exovetter import transit_coverage


def test_coverage1():
    time = np.arange(0, 100, step=.01)
    p_day = 3
    epoch = 3
    dur_hour = 5

    coverage, h ,b = transit_coverage.calc_coverage(
        time, p_day, epoch, dur_hour, ndur=2, nbins=10)

    assert coverage == 1.0


def test_coverage2():
    time = np.arange(0, 100, step=1)
    p_day = 3
    epoch = 3
    dur_hour = 12

    coverage, h ,b = transit_coverage.calc_coverage(
        time, p_day, epoch, dur_hour, ndur=2, nbins=10)

    assert coverage == 0.1


def test_coverage3():
    time = np.arange(0, 100, step=1)
    p_day = 5
    epoch = 10.25
    dur_hour = 24

    coverage, h,b = transit_coverage.calc_coverage(
        time, p_day, epoch, dur_hour, ndur=2, nbins=10)

    print(coverage)
    assert coverage == 0.2
