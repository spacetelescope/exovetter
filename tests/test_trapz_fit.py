"""Test trapezoid fit."""

import numpy as np
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

    def test_trapz_fit(self):
        noise_level = 40.0  # noise per observation in ppm
        data_series = 1.0 + np.random.randn(self.n_data) / 1e6 * noise_level
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
        # NOTE: Some results are non-deterministic when pytest run repeatedly?
        ioblk = TrapezoidFit.trapezoid_fit(
            self.time_series, data_series, error_series,
            self.signal_period, self.signal_epoch + 0.001,
            self.signal_duration_hours * 0.9, self.signal_depth * 1.1,
            fit_trial_n=2, fit_region=4.0, error_scale=1.0, sample_n=15)

        assert ioblk.minimized
        assert ioblk.chi2min > 400

        # To, Depth, BigT, TRatio
        assert_allclose(ioblk.bestphysvals, ioblk.physvals)
        assert_allclose(ioblk.bestboundedvals, ioblk.boundedvals)

        planet = ioblk.planetests
        assert_allclose(planet.u1, 0.4)
        assert_allclose(planet.u2, 0.27)
        assert_allclose(planet.period, 10.4203)
        assert_allclose(planet.radius_ratio, 0.017237, rtol=0.05)
        assert planet.semi_axis_ratio > 1
        assert planet.surface_brightness > 0.5
        assert planet.equiv_radius_ratio > 0
        assert planet.min_depth > 250
        assert_allclose(planet.epoch, 36.36210120066754, rtol=1e-4)
        assert planet.big_t > 0.2
        assert planet.little_t > 0
        assert planet.depth > 0.0002

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
