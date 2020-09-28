"""Utility functions."""

import sys
import warnings

import numpy as np

__all__ = ['sine', 'mark_transit_cadences', 'WqedLSF']


def sine(x, order, period=1):
    """Sine function for SWEET vetter."""
    w = 2 * np.pi / period
    if order == 0:
        return np.sin(w * x)
    elif order == 1:
        return np.cos(w * x)
    else:
        raise ValueError("Order should be zero or one")


def mark_transit_cadences(time, period_days, epoch_bkjd, duration_days,
                          num_durations=1, flags=None):
    """Create a logical array indicating which cadences are
    affected by a transit.

    Parameters
    ----------
    time : array_like
        Numpy 1D array of cadence times.

    period_days : float
        Transit period in days.

    epoch_bkjd : float
        Transit epoch.

    duration_days : float
        Duration of transit (start to finish). If you select a duration from
        first to last contact, all cadences affected by transit are selected.
        If you only select 2 to 3 contacts, only the interior region of the
        transit is selected.

    num_durations : int
        How much of the lightcurve on either side of the transit to mark.
        Default means to mark 1/2 the transit duration on either side of the
        transit center.

    flags : array_like or `None`
        If set, must be an array of booleans of length ``time``.
        Cadences where this array is `True` are ignored in the calculation.
        This is useful if some of the entries of time are NaN.

    Returns
    -------
    idx : array_like
        Array of booleans with the same length as ``time``.
        Element set to `True` are affected by transits.

    Raises
    ------
    ValueError
        Invalid input.

    """
    if flags is None:
        flags = np.zeros_like(time, dtype=np.bool_)

    good_time = time[~flags]
    i0 = np.floor((np.min(good_time) - epoch_bkjd) / period_days)
    i1 = np.ceil((np.max(good_time) - epoch_bkjd) / period_days)
    if not np.isfinite(i0) or not np.isfinite(i1):
        raise ValueError('Error bounding transit time')

    transit_times = epoch_bkjd + period_days * np.arange(i0, i1 + 1)
    big_num = sys.float_info.max  # A large value that isn't NaN
    max_diff = 0.5 * duration_days * num_durations

    idx = np.zeros_like(time, dtype=np.bool8)
    for tt in transit_times:
        diff = time - tt
        diff[flags] = big_num
        if not np.all(np.isfinite(diff)):
            raise ValueError('NaN found in diff of transit time')

        idx = np.bitwise_or(idx, np.fabs(diff) < max_diff)

    if not np.any(idx):
        warnings.warn('No cadences found matching transit locations')

    return idx


class WqedLSF:
    """Least squares fit to an analytic function based on ``lsf.c`` in Wqed,
    which in turn was based on Bevington and Robinson.

    Parameters
    ----------
    x : array_like
        1D numpy array containing ordinate (e.g., time).

    y : array_like
        1D numpy array containing coordinate (e.g., flux).

    s : array_like or float
        A scalar or 1D numpy array containing 1-sigma uncertainties.
        If not given, it is set to unity (default).

    order : int
        How many terms of function to fit. Default is 2.

    func : obj
        The analytic function to fit. Default is :func:`sine`.

    kwargs : dict
        Additional keywords to pass to ``func``.

    Raises
    ------
    ValueError
        Invalid input.

    """
    def __init__(self, x, y, s=None, order=2, func=sine, **kwargs):
        size = len(x)

        if func == sine and 'period' not in kwargs:
            raise ValueError("period must be provided for sine function")

        if size == 0:
            raise ValueError("x is zero length")
        if size != len(y):
            raise ValueError("x and y must have same length")

        if order < 1:
            raise ValueError("Order must be at least 1")
        if order >= size:
            raise ValueError("Length of input must be at least one greater "
                             "than order")

        if not np.all(np.isfinite(x)):
            raise ValueError("Nan or Inf found in x")
        if not np.all(np.isfinite(y)):
            raise ValueError("Nan or Inf found in y")

        if s is None:
            s = np.ones(size)
        elif np.isscalar(s):
            s = np.zeros(size) + s
        elif size != len(s):
            raise ValueError("x and s must have same length")
        elif not np.all(np.isfinite(s)):
            raise ValueError("Nan or Inf found in s")

        self.data_size = size
        self.x = x
        self.y = y
        self.s = s
        self.order = order
        self.func = func
        self.kwargs = kwargs

        self._fit()

    def _fit(self):
        """Fit the function to the data and return the best fit parameters."""

        df = np.empty((self.data_size, self.order))
        for i in range(self.order):
            df[:, i] = self.func(self.x, i, **self.kwargs)
            df[:, i] /= self.s

        A = np.matrix(df.transpose()) * np.matrix(df)
        covar = A.I  # Inverts

        wy = np.matrix(self.y / self.s)
        beta = wy * df
        params = beta * covar

        # Store results
        self._param = np.array(params)[0]  # Convert from matrix
        self._cov = covar

    def get_best_fit_model(self, x=None):
        """Get best fit model to data.

        Parameters
        ----------
        x : array_like
            1D numpy array containing ordinates on which to compute the
            best fit model. If not given, use ordinates used in fit.

        Returns
        -------
        y : array_like
            Best fit model.

        """
        if x is None:
            x = self.x

        par = self.params
        y = np.zeros_like(x)
        for i in range(self.order):
            y += par[i] * self.func(x, i, **self.kwargs)
        return y

    @property
    def params(self):
        """Best fit parameters."""
        return self._param

    @property
    def covariance(self):
        """Covariance matrix for the best fit."""
        return self._cov

    @property
    def residuals(self):
        """Residuals, defined as ``y - best_fit``."""
        return self.y - self.get_best_fit_model()

    @property
    def variance(self):
        """Sum of the squares of the residuals."""
        resid = self.residuals
        return np.sum(resid * resid) / (self.data_size - self.order)

    def compute_amplitude(self):
        """Amplitude from best fit.

        .. note::

            Taken from Appendix 1 of Breger (1999, A&A 349, 225), which was
            written by M Montgomery.

        Returns
        -------
        amp, amp_unc : float
            Amplitude and its uncertainty.

        Raises
        ------
        ValueError
            Fitted function is not sine with order 2.

        """
        if self.func != sine or self.order != 2:
            raise ValueError('Only applicable to sine function with order=2')

        par = self.params
        amp = np.hypot(par[0], par[1])
        amp_unc = np.sqrt(2 * self.variance / self.data_size)

        return amp, amp_unc

    def compute_phase(self):
        """Phase from best fit.

        .. note::

            Taken from Appendix 1 of Breger (1999, A&A 349, 225), which was
            written by M Montgomery.

        Returns
        -------
        phase, phase_unc : float
            Phase and its uncertainty.

        Raises
        ------
        ValueError
            Fitted function is not sine with order 2.

        """
        par = self.params
        amp, amp_unc = self.compute_amplitude()

        # r1 is cos(phase), r2 is -sin(phase) (because fitting wx-phi)
        r1 = par[0] / amp
        r2 = -par[1] / amp

        # Remember the sign of the cos and sin components
        invcos = np.arccos(np.fabs(r1))
        invsin = np.arcsin(np.fabs(r2))

        # Decide what quadrant we're in.
        # First quadrant is no-op: (r1>=0 and r2 >=0)
        if r1 <= 0 and r2 >= 0:  # Second quadrant
            invcos = np.pi - invcos
            invsin = np.pi - invsin
        elif r1 <= 0 and r2 <= 0:  # Third quadrant
            invcos += np.pi
            invsin += np.pi
        elif r1 >= 0 and r2 <= 0:  # Fourth quadrant
            invcos = 2 * np.pi - invcos
            invsin = 2 * np.pi - invsin

        # Choose the average of the two deteminations to
        # reduce effects of roundoff error
        phase = 0.5 * (invcos + invsin)

        return phase, amp_unc / amp
