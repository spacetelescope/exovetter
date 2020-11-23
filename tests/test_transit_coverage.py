import numpy as np
import pytest
from numpy.testing import assert_allclose

from exovetter import transit_coverage


@pytest.mark.parametrize(
    ('step', 'p_day', 'epoch', 'dur_hour', 'ans'),
    [(0.01, 3, 3, 5, 1),
     (1, 3, 3, 12, 0.1),
     (1, 5, 10.25, 24, 0.2)])
def test_coverage1(step, p_day, epoch, dur_hour, ans):
    time = np.arange(0, 100, step=step)
    coverage = transit_coverage.calc_coverage(
        time, p_day, epoch, dur_hour, ndur=2, nbins=10)
    assert_allclose(coverage, ans)
