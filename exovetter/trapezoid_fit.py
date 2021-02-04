# Originally
# https://github.com/exoplanetvetting/DAVE/blob/master/trapezoidFit/trapfit.py
"""Module to perform a trapezoid model fit to flux time series data.

Originally written by Christopher Burke [1]_.

References
----------
.. [1] Kostov, V. B., Mullaly, S. E., Quintana, E. V. et al. 2019, AJ, 157, 3
    (arXiv:1901.07459)

"""
import os
import sys

import numpy as np

__all__ = ['phase_data', 'trapezoid_vectors', 'TrapezoidFitParameters',
           'TrapezoidOriginalEstimates', 'TrapezoidPlanetEstimates',
           'TrapezoidFit']


def phase_data(t, period, t_zero):
    """Phase the data at period per and centered at to inputs given.

    Parameters
    ----------
    t : array_like
       Time of data.

    period : array_like
        Period to phase time data.
        ``period`` and ``t`` should be in the same units.

    t_zero: array_like
        Epoch of phase zero.

    Returns
    -------
    phi : array_like
        Data phased running from ``(-0.5, 0.5]``.

    """
    phi = np.mod(t - t_zero, period) / period
    return np.where(phi > 0.5, phi - 1.0, phi)


def trapezoid_vectors(t, depth, big_t, little_t):
    """Trapezoid shape, in the form of vectors, for model.

    Parameters
    ----------
    t : float
        Vector of independent values to evaluate trapezoid model.

    depth : float
        Depth of trapezoid.

    big_t : float
        Full trapezoid duration.

    little_t : float
        Ingress/egress duration.

    Returns
    -------
    output : float
        Vector of trapezoid model values.

    """
    output = np.full_like(t, 1.0)
    t = np.abs(t)
    big_t_half = big_t * 0.5
    little_t_half = little_t * 0.5
    one_minus_depth = 1.0 - depth
    output = np.where(t <= big_t_half - little_t_half, one_minus_depth, output)
    return np.where(
        np.logical_and(t > big_t_half - little_t_half,
                       t < big_t_half + little_t_half),
        one_minus_depth + ((depth / little_t) *
                           (t - big_t_half + little_t_half)),
        output)


class TrapezoidFitParameters:
    """Class to handle the parameters of the trapezoid fit algorithms.

    Parameters
    ----------
    cadlen : float
        Cadence duration in days.

    samplen : int
        Subsampling of light curve model data. Must be odd integer.

    fitregion : float
        Factor of duration around midpoint for data fitting.

    """

    def __init__(self, cadlen, samplen=15, fitregion=4.0):
        self.cadlen = cadlen
        self.samplen = samplen
        self.fitregion = fitregion

    @property
    def samplen(self):
        """Subsampling of light curve model data. Must be odd integer."""
        return self._samplen

    @samplen.setter
    def samplen(self, val):
        val = int(val)
        if val % 2 != 1:
            raise ValueError(f'samplen must be odd but {val} is given')
        self._samplen = val

    def __str__(self):
        return os.linesep.join(
            [self.__class__.__name__,
             f'cadlen = {self.cadlen}',
             f'samplen = {self.samplen}',
             f'fitregion = {self.fitregion}'])


class TrapezoidOriginalEstimates:
    """Class to handle the original parameter estimations.

    Parameters
    ----------
    period : float
        Initial orbital period in days.
        **This is adjusted during fitting.**

    epoch : float
        Initial epoch in days.

    duration : float
        Initial duration fitted **in hours**.

    depth : float
        Initial depth in ppm.

    """

    def __init__(self, period=1.0, epoch=0.1, duration=3.0, depth=100.0):
        self.period = period
        self.epoch = epoch
        self.duration = duration
        self.depth = depth

    def __str__(self):
        return os.linesep.join(
            [self.__class__.__name__,
             f'period = {self.period}',
             f'epoch = {self.epoch}',
             f'duration = {self.duration}',
             f'depth = {self.depth}'])


class TrapezoidPlanetEstimates:
    """Class to handle estimating a planet's model based upon
    the trapezoid fit solution.
    See Carter et al. 2008.

    Parameters
    ----------
    u1, u2 : float
        Quadratic limb darkening parameters.
        The defaults are the limb darkening for Sun in the Kepler passband.

    period : float
        Resulting period in days. Currently not fit.

    radius_ratio : float
        Radius ratio from purely geometric depth, defined as ``(Rp/Rstar)^2``.

    impact_parameter : float
        Impact parameter.

    tauzero : float
        Transit timescale constant in days.

    semi_axis_ratio : float
        Semi-major axis to stellar radius ratio.

    surface_brightness : float
        Limb darkened surface brightness at crossing impact parameter.

    equiv_radius_ratio : float
        Crude approximation to radius ratio, taking into account limb darkening
        that works better than the purely geometric radius ratio.

    min_depth : float
        Minimum depth from model in ppm.

    epoch : float
        Epoch of fit midpoint.

    big_t : float
        Trapezoid model full duration in days.

    little_t : float
        Trapezoid model ingress/egress duration in days.

    depth : float
        Trapezoid model depth parameter in ppm.

    """

    def __init__(self, u1=0.4, u2=0.27, period=1.0, radius_ratio=0.0,
                 impact_parameter=0.5, tauzero=0.1, semi_axis_ratio=20.0,
                 surface_brightness=0.5, equiv_radius_ratio=0.0, min_depth=0.0,
                 epoch=0.0, big_t=0.0, little_t=0.0, depth=0.0):
        self.u1 = u1
        self.u2 = u2
        self.period = period
        self.radius_ratio = radius_ratio
        self.impact_parameter = impact_parameter
        self.tauzero = tauzero
        self.semi_axis_ratio = semi_axis_ratio
        self.surface_brightness = surface_brightness
        self.equiv_radius_ratio = equiv_radius_ratio
        self.min_depth = min_depth
        self.epoch = epoch
        self.big_t = big_t
        self.little_t = little_t
        self.depth = depth

    def __str__(self):
        return os.linesep.join(
            [self.__class__.__name__,
             f'u1 = {self.u1}',
             f'u2 = {self.u2}',
             f'period = {self.period}',
             f'radius_ratio = {self.radius_ratio}',
             f'impact_parameter = {self.impact_parameter}',
             f'tauzero = {self.tauzero}',
             f'semi_axis_ratio = {self.semi_axis_ratio}',
             f'surface_brightness = {self.surface_brightness}',
             f'equiv_radius_ratio = {self.equiv_radius_ratio}',
             f'min_depth = {self.min_depth}',
             f'epoch = {self.epoch}',
             f'big_t = {self.big_t}',
             f'little_t = {self.little_t}',
             f'depth = {self.depth}'])


class TrapezoidFit:
    """Class to handle trapezoid fits.

    Parameters
    ----------
    time_series, data_series, error_series : array_like
        Data, uncertainty, and time series.

    trp_parameters : `TrapezoidFitParameters` or `None`
        Trapezoid fit parameters. If not given, defaults are used.

    trp_originalestimates : `TrapezoidOriginalEstimates` or `None`
        Trapezoid fit original estimates. If not given, defaults are used.

    trp_planetestimates : `TrapezoidOriginalEstimates` or `None`
        Trapezoid fit planet model estimates. If not given, defaults are used.

    t_ratio : float
        Value for ``TRatio`` in ``physvals``.

    error_scale : float
        Errorbars scaling for :meth:`trp_likehood` residual calculation.

    logger : obj
        Python logger. If not given, default logger is used.

    Attributes
    ----------
    logger : obj
        Logger.

    parm : `TrapezoidFitParameters`
        Trapezoid fit parameters.

    origests : `TrapezoidOriginalEstimates`
        Trapezoid fit original estimates.

    planetests : `TrapezoidPlanetEstimates`
        Planet model estimates to be set in :meth:`trp_estimate_planet`.

    normlc : array_like
        Data series (light curve).

    normes : array_like
        Uncertainty of data series.

    normots : array_like
        Original time series.

    timezpt : float
        Zeropoint of time series.

    normts : array_like
        Normalized time series.

    sampleit : array_like
        Region for over-sampling.

    fitdata : array_like
        Region for fitting.

    physvals : array_like
        Values corresponding to `physval_names`.

    physval_mins : array_like
        Minimum limits for ``physvals``.

    physval_maxs : array_like
        Maximum limits for ``physvals``.

    physvalsavs : array_like
        Store ``physvals`` parameters that are fixed during the calculation.

    bestphysvals : array_like
        Best ``physvals`` to be calculated in :meth:`trp_iterate_solution`.

    boundedvals : array_like
        Bounded version of ``physvals`` set in :meth:`set_boundedvals`.

    boundedvalsavs : array_like
        Store ``boundedvals`` parameters that are fixed during the calculation.

    bestboundedvals : array_like
        Best ``boundedvals`` to be calculated in :meth:`trp_iterate_solution`.

    fixed : array_like
        Flags for whether a ``physvals`` element is fixed during fitting.

    nparm : int
        Number of elements in ``physvals``.

    modellc : array_like
        Trapezoid model to be set in :meth:`trapezoid_model`.

    error_scale : float
        Errorbars scaling for :meth:`trp_likehood` residual calculation.

    chi2min : float
        Best chi-squared to be set in :meth:`trp_iterate_solution`.

    likecount : int
        Number of times :meth:`trp_likehood` was called.

    minimized : bool
        Flag of whether fitting has occured or not.

    """

    def __init__(self, time_series, data_series, error_series,
                 trp_parameters=None, trp_originalestimates=None,
                 trp_planetestimates=None, t_ratio=0.2, error_scale=1.0,
                 logger=None):
        if logger is None:
            import logging
            self.logger = logging.getLogger('TrapezoidFit')
        else:
            self.logger = logger

        if trp_parameters:
            self.parm = trp_parameters
        else:
            # FIXME: This is Kepler cadence, use TESS.
            self.parm = TrapezoidFitParameters(29.424 / 60.0 / 24.0)

        if trp_originalestimates:
            self.origests = trp_originalestimates
        else:
            self.origests = TrapezoidOriginalEstimates()

        if trp_planetestimates:
            self.planetests = trp_planetestimates
        else:
            self.planetests = TrapezoidPlanetEstimates()

        self.normlc = data_series
        self.normes = error_series
        self.normots = time_series
        self.error_scale = error_scale

        per = self.origests.period
        eph = self.origests.epoch

        # Normalize the time series
        median_event = np.median(np.round((self.normots - eph) / per))
        self.timezpt = eph + (median_event * per)
        self.normts = self.normots - self.timezpt

        dur = self.origests.duration
        depth = self.origests.depth / 1.0e6
        durday = dur / 24.0
        phidur = dur / 24.0 / per

        # Identify in transit data to over sample and fitting region
        phi = phase_data(self.normts, per, 0.0)
        self.sampleit = np.where(
            abs(phi) < (phidur * 1.5), self.parm.samplen, 1)
        self.fitdata = np.where(
            abs(phi) < (phidur * self.parm.fitregion), True, False)

        # Always fit less than a 0.25 of phase space for stability
        # and efficiency reasons
        self.fitdata = np.where(abs(phi) > 0.25, False, self.fitdata)

        # Set parameters and bounds
        self.physvals = np.array([0.0, depth, durday, t_ratio])
        self.physval_mins = np.array(
            [-durday * 1.5, 1.0e-6, 0.0, 1.0e-10])
        self.physval_maxs = np.array(
            [durday * 1.5, depth * 5.0, durday * 3.0, 1.0])
        self.fixed = np.array([0, 0, 0, 0])
        self.nparm = np.size(self.fixed)

        self.modellc = np.full_like(self.normlc, 1.0)
        self.chi2min = self.normlc.size * 2000.0

        self.set_boundedvals()  # Set self.boundedvals

        # Used to store parameters that are fixed during the calculation.
        # They must be populated with fixed values before moving forward.
        self.physvalsavs = self.physvals
        self.boundedvalsavs = self.boundedvals

        self.bestphysvals = self.physvals
        self.bestboundedvals = self.boundedvals

        self.likecount = 0
        self.minimized = False

        self.trp_validate()

    @property
    def physval_names(self):
        """Description of stored `physval`` values."""
        return ('To', 'Depth', 'BigT', 'TRatio')

    def trp_validate(self):
        """Validate that trapezoid fit inputs look reasonable.

        Raises
        ------
        ValueError
            Validation failed.

        """
        err_msgs = []

        # Check that physvals are within limits
        if (np.any(np.greater_equal(self.physvals, self.physval_maxs))):
            err_msgs.append(f'physvals: {self.physvals} is greater than '
                            f'physval_maxs: {self.physval_maxs}')
        if (np.any(np.less_equal(self.physvals, self.physval_mins))):
            err_msgs.append(f'physvals: {self.physvals} is less than '
                            f'physval_mins: {self.physval_mins}')
        # Check for NaNs in input data series
        if (np.any(np.isnan(self.normlc))):
            err_msgs.append('Input light curve contains NaN')
        if (np.any(np.isnan(self.normes))):
            err_msgs.append('Input uncertainty estimate contains NaN')
        if (np.any(np.isnan(self.normots))):
            err_msgs.append('Input time data contains NaN')
        # Check for input data series that has negative flux data should be
        # normalized to 1.0
        if (np.any(np.less(self.normlc, 0.0))):
            err_msgs.append('Negative flux in light curve')

        if err_msgs:
            raise ValueError(f'Validation failed:{os.linesep}'
                             f'{os.linesep.join(err_msgs)}')

    def set_boundedvals(self):
        """Convert parameters to bounded versions that the minimizer will use.

        This sets ``self.boundedvals``.

        Raises
        ------
        ValueError
            Bounded values are invalid.

        """
        maxmindelta = self.physval_maxs - self.physval_mins
        datamindelta = self.physvals - self.physval_mins
        self.boundedvals = -np.log(maxmindelta / datamindelta - 1.0)
        if ~np.isfinite(self.boundedvals).all():
            raise ValueError('Invalid bounded values:\n'
                             f'boundedvals = {self.boundedvals}\n'
                             f'physvals = {self.physvals}')

    def set_physvals(self):
        """Convert bounded parameter values that the minimizer uses to
        physical values.

        This sets ``self.physvals``.

        Raises
        ------
        ValueError
            Physical values are invalid.

        """
        maxmindelta = self.physval_maxs - self.physval_mins
        self.physvals = self.physval_mins + (
            maxmindelta / (1.0 + np.exp(-self.boundedvals)))
        if ~np.isfinite(self.physvals).all():
            raise ValueError('Invalid physical values:\n'
                             f'boundedvals = {self.boundedvals}\n'
                             f'physvals = {self.physvals}')

    @classmethod
    def trapezoid_model_onemodel(cls, ts, period, epoch, depth, big_t,
                                 little_t, subsamplen):
        """Make a trapezoid model at the given input parameters.

        You can save time if you want to generate many models by
        calling this once to generate the instance and then call
        :meth:`trapezoid_model_raw` to generate the models at other inputs,
        thus bypassing some of the setup routines.

        Parameters
        ----------
        ts : array_like
            Mid-cadence time stamps.

        period : float
            Period of signal.
            **Assumed fixed during model generation.**

        epoch : float
            Estimated epoch of signal. Must be on the same system as ``ts``.

        depth : float
            Model depth in ppm.

        big_t : float
            Full transit duration in hours.

        little_t : float
            Ingress time in hours.

        subsamplen : int
            Subsample each cadence by this factor.

        Returns
        -------
        ioblk : `TrapezoidFit`
            Instance that is used in the transit model.
            It has model light curve in its ``modellc`` attribute.

        """
        # Calculate this from timeSeries
        cadlen = np.median(np.diff(ts))

        dummy = np.array([0.0])
        trp_parm = TrapezoidFitParameters(cadlen, samplen=subsamplen)
        trp_origests = TrapezoidOriginalEstimates(
            period=period, epoch=epoch, duration=big_t, depth=depth)
        ioblk = cls(ts, dummy, dummy, trp_parameters=trp_parm,
                    trp_originalestimates=trp_origests,
                    t_ratio=(little_t / big_t))
        ioblk.trapezoid_model()

        return ioblk

    def trapezoid_model_raw(self, epoch, depth, big_t, little_t):
        """Generate another model at a different epoch depth duration
        and ingress time.

        Run this after you have a pre-existing `TrapezoidFit` instance
        from fit or from :meth:`trapezoid_model_onemodel`.

        .. note:: Period is not variable at this point.

        Parameters
        ----------
        epoch : float
            Estimated epoch of signal. Must be on the same system time series.

        depth : float
            Model depth in ppm.

        big_t : float
            Full transit duration in hours.

        little_t : float
            Ingress time in hours.

        Returns
        -------
        ioblk : `TrapezoidFit`
            New instance containing model light curve in its ``modellc``
            attribute.

        """
        from copy import deepcopy
        ioblk = deepcopy(self)
        ioblk.physvals[0] = epoch - ioblk.origests.epoch
        ioblk.physvals[1] = depth / 1.0e6
        ioblk.physvals[2] = big_t / 24.0
        ioblk.physvals[3] = little_t / big_t
        ioblk.set_boundedvals()
        ioblk.trapezoid_model()
        return ioblk

    def trapezoid_model(self):
        """Generate a subsampled model at the current parameters.

        This sets ``self.modellc``.

        Raises
        ------
        ValueError
            Model generation failed.

        """
        to = self.physvals[0]
        depth = self.physvals[1]
        big_t = self.physvals[2]
        little_t = self.physvals[3] * big_t
        per = self.origests.period
        ts = self.normts
        phi = phase_data(ts, per, to)
        lc = np.ones_like(self.normts)
        cadlen = self.parm.cadlen
        samplen = self.parm.samplen

        # Call trapezoid model for data points without any subsampling needed
        idx = np.where(np.logical_and(self.fitdata, self.sampleit == 1))[0]
        if idx.size > 0:
            ztmp = phi[idx] * per
            lctmp = trapezoid_vectors(ztmp, depth, big_t, little_t)
            lc[idx] = lctmp

        # Call trapezoid model for data points that need subsampling
        idx = np.where(np.logical_and(self.fitdata, self.sampleit > 1))[0]
        if idx.size > 0:
            ztmp = phi[idx] * per
            cadlen_div2 = cadlen * 0.5
            delta_x_small_div2 = cadlen_div2 / float(samplen)
            small_blk = np.linspace(-cadlen_div2 + delta_x_small_div2,
                                    cadlen_div2 - delta_x_small_div2, samplen)
            o_n = ztmp.size
            ztmp_highres = np.tile(ztmp, samplen)
            ztmp_highres = np.reshape(ztmp_highres, (samplen, o_n))
            small_blk_highres = np.tile(small_blk, o_n)
            small_blk_highres = np.reshape(small_blk_highres, (o_n, samplen))
            small_blk_highres = np.transpose(small_blk_highres)
            ztmp_highres += small_blk_highres
            ztmp_highres = ztmp_highres.ravel(order='F')
            lctmp_highres = trapezoid_vectors(
                ztmp_highres, depth, big_t, little_t)
            n_n = ztmp_highres.size
            lctmp = lctmp_highres.reshape([o_n, int(n_n / o_n)]).mean(1)
            lc[idx] = lctmp

        self.modellc = lc
        if np.sum(np.isfinite(lc)) != lc.size:
            raise ValueError('Trapezoid model generation failed')

    def trp_likehood(self, pars):
        """Likelihood function used for Scipy optimizer.
        Some attributes are modified in-place as well.

        Parameters
        ----------
        pars : array_like
            Vector of parameter values.

        Returns
        -------
        residuals : array_like
            Sum of squares of residuals of ``data - model``.

        """
        self.likecount += 1
        # Update parameters into bounded values
        idx = np.where(self.fixed == 0)[0]
        self.boundedvals[idx] = pars
        self.boundedvals = np.where(self.fixed == 1, self.boundedvalsavs,
                                    self.boundedvals)
        # Convert to unbounded values
        self.set_physvals()
        # Generate Model
        self.trapezoid_model()
        # Calculate residuals
        idx = np.where(self.fitdata)[0]
        residuals = ((self.normlc[idx] - self.modellc[idx]) /
                     (self.normes[idx] * self.error_scale))
        # Return scalar summed residuals
        return np.sum(residuals**2)

    # TODO: Not sure what this was for but plotting should be optional,
    # so this was taken out of the function being passed into scipy optimize.
    # Fix as needed.
    def plot_likehood(self, show_legend=False):
        """Plot results from :meth:`trp_likehood`."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(3, 2), dpi=300)
        ax = fig.subplots()
        ax.set_position([0.125, 0.125, 0.825, 0.825])

        period = self.origests.period
        tzero = self.physvals[0]
        ts = self.normts
        phi = phase_data(ts, period, tzero)

        ax.plot(phi, self.normlc, '.', markersize=0.6, label='input')
        ax.plot(phi, self.modellc, '.r', markersize=0.6, label='model')
        if show_legend:
            ax.legend()
        plt.draw()

    def trp_iterate_solution(self, n_iter=20, method='Powell', options=None,
                             seed=None):
        """Perform multiple iterations starting from random initial conditions.
        Attributes are updated in-place with the best solution in a chi-squared
        sense among the iterations.

        Parameters
        ----------
        n_iter : int
            Number of iterations to run.

        method : str
            See :func:`scipy.optimize.minimize`.

        options : dict or `None`
            See :func:`scipy.optimize.minimize`.
            If not given, some pre-defined defaults are used.

        seed : int or `None`
            Seed for Numpy random generator.
            Usually used in testing for reproducibility.

        """
        import scipy.optimize as opt
        from astropy.utils.compat.context import nullcontext
        from astropy.utils.misc import NumpyRNGContext

        if options is None:
            options = {'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 2000,
                       'maxfev': 2000}

        if seed is None:
            rand_ctx = nullcontext()
        else:
            rand_ctx = NumpyRNGContext(seed)

        best_chi2s = np.zeros(n_iter)
        best_pars = np.zeros((self.physvals.size, n_iter))
        gd_fits = np.zeros(n_iter, dtype=bool)
        depth_half = self.origests.depth * 0.5 / 1.0e6
        depth_half_abs = np.abs(depth_half)

        with rand_ctx:
            for i in range(n_iter):
                self.physvals = (self.physval_mins +
                                 np.random.rand(self.physvals.size) *
                                 (self.physval_maxs - self.physval_mins))
                # Force depth parameter to start at minimum half the depth
                if self.physvals[1] < depth_half_abs:
                    self.physvals[1] = depth_half

                # Replace random starts with parameters values that are fixed
                self.physvals = np.where(self.fixed == 1, self.physvalsavs,
                                         self.physvals)
                self.set_boundedvals()
                freeidx = np.where(self.fixed == 0)[0]
                start_pars = self.boundedvals[freeidx]

                all_output = opt.minimize(self.trp_likehood, start_pars,
                                          method=method, options=options)
                self.boundedvals[freeidx] = all_output['x']
                self.boundedvals = np.where(
                    self.fixed == 1, self.boundedvalsavs, self.boundedvals)
                self.set_physvals()
                chi2min = all_output['fun']
                self.logger.debug(f'It: {i}  Chi2: {chi2min}\n{self.physvals}')

                if np.isfinite(self.physvals).all():
                    gd_fits[i] = True
                    best_chi2s[i] = chi2min
                    best_pars[:, i] = self.physvals

        # Done with iterations find the best one by chi2min
        best_masked_idx = np.argmin(best_chi2s[gd_fits])
        self.chi2min = best_chi2s[gd_fits][best_masked_idx]
        self.bestphysvals = best_pars[:, gd_fits][:, best_masked_idx]
        self.physvals = self.bestphysvals
        self.set_boundedvals()
        self.bestboundedvals = self.boundedvals
        self.logger.debug(
            f'Overall Best Chi2 Min: {self.chi2min}\n{self.physvals}')
        self.minimized = True

    def trp_estimate_planet(self):
        """Perform crude estimate of a planet model that is close to
        the trapezoid solution, which must be present for this to work.

        This mainly sets ``self.planetests``.

        """
        if not self.minimized:
            raise ValueError("Getting planet estimates for non-converged"
                             "trapezoid fit is not allowed")

        self.planetests.period = self.origests.period
        self.planetests.epoch = self.timezpt + self.bestphysvals[0]
        self.planetests.big_t = self.bestphysvals[2]
        self.planetests.little_t = self.bestphysvals[3] * self.planetests.big_t
        self.planetests.depth = self.bestphysvals[1]

        # Call likehood to get best transit model
        idx = np.where(self.fixed == 0)[0]
        self.trp_likehood(self.bestboundedvals[idx])
        trapmodlc = self.modellc

        self.planetests.min_depth = (1.0 - trapmodlc.min()) * 1.0e6
        self.planetests.radius_ratio = np.sqrt(self.planetests.min_depth / 1e6)
        self.planetests.impact_parameter = np.sqrt(
            1.0 - np.amin([self.planetests.radius_ratio *
                           self.planetests.big_t / self.planetests.little_t,
                           1.0]))
        self.planetests.tauzero = np.sqrt(
            self.planetests.big_t * self.planetests.little_t / 4.0 /
            self.planetests.radius_ratio)
        self.planetests.semi_axis_ratio = (
            self.planetests.period / 2.0 / np.pi / self.planetests.tauzero)
        mu = np.sqrt(1.0 - self.planetests.impact_parameter ** 2)
        self.planetests.surface_brightness = (
            1.0 - self.planetests.u1 * (1.0 - mu) -
            self.planetests.u2 * (1.0 - mu) ** 2)
        self.planetests.equiv_radius_ratio = (
            self.planetests.radius_ratio /
            np.sqrt(self.planetests.surface_brightness))

    @classmethod
    def trapezoid_fit(cls, time_series, data_series, error_series,
                      signal_period, signal_epoch, signal_duration,
                      signal_depth, fit_trial_n=13, fit_region=4.0,
                      error_scale=1.0, sample_n=15, seed=None):
        """Perform a trapezoid fit to a normalized flux time series.
        Assumes all data has the same cadence duration.
        Period is fixed during the trapezoid fitting.

        Parameters
        ----------
        time_series : array_like
            Mid cadence time stamps.

        data_series : array_like
            Normalized time series.

        error_series : array_like
            Uncertainty (error) time series.

        signal_period : float
            Period of signal in days.
            **Assumed fixed during model fitting.**

        signal_epoch : float
            Estimated epoch of signal in days.
            Must be on the same system as ``time_series``.

        signal_duration : float
            Estimated signal duration **in hours**.

        signal_depth : float
            Estimated signal depth in ppm.

        fit_trial_n : int
            How many trial fits to perform starting at random initial
            locations. Increase this if the minimization is returning
            local minima.

        fit_region : float
            Fit data within ``fit_region * signal_duration`` of
            ``signal_epoch``.

        error_scale : float
            Scale the errorbars by this factor.

        sample_n : int
            Subsample each cadence by this factor.

        seed : int or `None`
            Seed for Numpy random generator.
            Usually used in testing for reproducibility.

        Returns
        -------
        ioblk : `TrapezoidFit`
            Instance storing all the information pertaining to fit results.

        """
        # Calculate this from time_series
        cadlen = np.median(np.diff(time_series))

        trp_parm = TrapezoidFitParameters(
            cadlen, samplen=sample_n, fitregion=fit_region)
        trp_origests = TrapezoidOriginalEstimates(
            period=signal_period, epoch=signal_epoch, duration=signal_duration,
            depth=signal_depth)
        ioblk = cls(
            time_series, data_series, error_series,
            trp_parameters=trp_parm, trp_originalestimates=trp_origests,
            error_scale=error_scale)

        # Find solution by trying random initial conditions
        ioblk.trp_iterate_solution(n_iter=fit_trial_n, seed=seed)

        # Convert the trapezoid fit solution into a pseudo planet model
        # parameters
        ioblk.trp_estimate_planet()

        # Raise an exception if final model is consistent with flat
        if (np.all(np.abs(ioblk.modellc - ioblk.modellc[0])
                   < (10.0 * sys.float_info.epsilon))):
            raise ValueError('Model light curve is flat')

        # Check for NaNs in output model
        if (np.any(np.isnan(ioblk.modellc))):
            raise ValueError('Output model light curve contains NaN')

        return ioblk
