import numpy as np
from exovetter import transit_coverage
import pytest
from numpy.testing import assert_allclose

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

@pytest.mark.parametrize(
    ('step', 'p_day', 'epoch', 'dur_hour', 'ans'),
    [(0.01, 3, 3, 5, 1),
     (1, 3, 3, 12, 0.1),
     (1, 5, 10.25, 24, 0.2)])

def test_coverage3(step, p_day, epoch, dur_hour, ans):
    time = np.arange(0, 100, step=step)
    coverage, h, b = transit_coverage.calc_coverage(
        time, p_day, epoch, dur_hour, ndur=2, nbins=10)
    assert_allclose(coverage, ans)