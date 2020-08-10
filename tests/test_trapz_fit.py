"""Test trapezoid fit."""

import numpy as np

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
        data_series = data_series * ioblk.modellc

        # Test fitting
        ioblk = TrapezoidFit.trapezoid_fit(
            self.time_series, data_series, error_series,
            self.signal_period, self.signal_epoch + 0.001,
            self.signal_duration_hours * 0.9, self.signal_depth * 1.1,
            fit_trial_n=2, fit_region=4.0, error_scale=1.0, sample_n=15)

        # TODO: Some asserts?

    def test_model_gen(self):
        """Test generating model."""
        newioblk = TrapezoidFit.trapezoid_model_onemodel(
            self.time_series, self.signal_period,
            self.signal_epoch, self.signal_depth, self.signal_duration_hours,
            self.signal_duration_hours * 0.1, 15)

        newioblk = newioblk.trapezoid_model_raw(
            self.signal_epoch + 0.05, self.signal_depth * 1.5,
            self.signal_duration_hours * 2.0,
            self.signal_duration_hours * 2.0 * 0.2)

        # TODO: Some asserts?
