"""Test trapezoid fit."""

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from numpy.testing import assert_allclose

from exovetter.trapezoid_fit import (
    TrapezoidFitParameters, TrapezoidOriginalEstimates, TrapezoidFit)


class TestDAVE:
    """Adapted from DAVE test case."""

    def setup_class(self):
        """Make some fake data."""
        data_span = 80.0  # in Days
        self.exposure_length = 1.0 / 48.0  # in Days, 48 cadences per day
        self.n_data = int(data_span / self.exposure_length)
        self.signal_depth = 300.0  # signal depth in ppm
        signal_duration = 5.0 / 24.0  # in Days
        self.signal_duration_hours = signal_duration * 24.0
        self.signal_period = 10.4203  # in Days
        self.signal_epoch = 5.1  # in Days
        self.time_series = np.linspace(0, int(data_span), self.n_data)
        self.seed = 1234  # For reproducibility

    def test_trapz_fit(self):
        noise_level = 40.0  # noise per observation in ppm
        with NumpyRNGContext(self.seed):
            data_series = (1.0 +
                           np.random.randn(self.n_data) / 1e6 * noise_level)
        error_series = np.full_like(data_series, noise_level / 1e6)

        trp_parm = TrapezoidFitParameters(
            self.exposure_length, samplen=15, fitregion=4.0)
        trp_origests = TrapezoidOriginalEstimates(
            period=self.signal_period, epoch=self.signal_epoch,
            duration=self.signal_duration_hours, depth=self.signal_depth)

        # Instantiate class and fill in values
        ioblk = TrapezoidFit(
            self.time_series, data_series, error_series,
            trp_parameters=trp_parm, trp_originalestimates=trp_origests,
            t_ratio=0.1)

        # Make a model trapezoid light curve
        ioblk.trapezoid_model()

        # Insert signal
        data_series *= ioblk.modellc

        # Test fitting
        ioblk = TrapezoidFit.trapezoid_fit(
            self.time_series, data_series, error_series,
            self.signal_period, self.signal_epoch + 0.001,
            self.signal_duration_hours * 0.9, self.signal_depth * 1.1,
            fit_trial_n=2, fit_region=4.0, error_scale=1.0, sample_n=15,
            seed=self.seed)

        assert ioblk.minimized
        assert_allclose(ioblk.chi2min, 519.1195391259582)

        # To, Depth, BigT, TRatio
        assert_allclose(ioblk.physvals, [-0.00043501, 0.00030478,
                                         0.20759337,  0.03783464], rtol=5e-6)
        assert_allclose(ioblk.bestphysvals, ioblk.physvals)
        assert_allclose(ioblk.boundedvals, [-3.09340749e-03, -1.48800535,
                                            -5.36273499e-01, -3.23596121])
        assert_allclose(ioblk.bestboundedvals, ioblk.boundedvals)

        planet = ioblk.planetests
        assert_allclose(planet.u1, 0.4)
        assert_allclose(planet.u2, 0.27)
        assert_allclose(planet.period, 10.4203)
        assert_allclose(planet.radius_ratio, 0.017457976019387363)
        assert_allclose(planet.impact_parameter, 0.7338744124096862)
        assert_allclose(planet.tauzero, 0.15280281516406793)
        assert_allclose(planet.semi_axis_ratio, 10.853479706638709)
        assert_allclose(planet.surface_brightness, 0.8439424093636225)
        assert_allclose(planet.equiv_radius_ratio, 0.019003670291116378)
        assert_allclose(planet.min_depth, 304.7809266935042)
        assert_allclose(planet.epoch, 36.36146498991824)
        assert_allclose(planet.big_t, 0.20759337359154925)
        assert_allclose(planet.little_t, 0.007854220840564521)
        assert_allclose(planet.depth, 0.0003047809266932439)

    def test_model_gen(self):
        """Test generating model."""

        newioblk = TrapezoidFit.trapezoid_model_onemodel(
            self.time_series, self.signal_period,
            self.signal_epoch, self.signal_depth, self.signal_duration_hours,
            self.signal_duration_hours * 0.1, 15)

        newioblk2 = newioblk.trapezoid_model_raw(
            self.signal_epoch + 0.05, self.signal_depth * 1.5,
            self.signal_duration_hours * 2.0,
            self.signal_duration_hours * 2.0 * 0.2)

        assert len(newioblk.modellc) == 3840
        assert_allclose(newioblk.parm.cadlen, 0.020838760093774056)
        assert_allclose(newioblk.physvals, [0, 0.0003, 0.20833333, 0.1])
        assert_allclose(newioblk.boundedvals,
                        [-0, -1.38963326, -0.69314718, -2.19722458])

        assert len(newioblk2.modellc) == 3840
        assert_allclose(newioblk2.parm.cadlen, 0.020838760093774056)
        assert_allclose(newioblk2.physvals, [0.05, 0.00045, 0.41666667, 0.2])
        assert_allclose(newioblk2.boundedvals,
                        [0.32277339, -0.84952256, 0.69314718, -1.38629436])
