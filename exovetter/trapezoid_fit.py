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
            [f'cadlen = {self.cadlen}',
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
            [f'period = {self.period}',
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
            [f'u1 = {self.u1}',
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


# TODO: UNTIL HERE

class TrapezoidFit:
    """Class to handle trapezoid fits."""

    # TODO: let user pass in stuff in init?
    def __init__(self, trp_parameters=None, trp_originalestimates=None,
                 trp_planetestimates=None, logger=None):
        if logger is None:
            import logging
            self.logger = logging.getLogger('TrapezoidFit')
        else:
            self.logger = logger

        if trp_parameters:
            self.parm = trp_parameters
        else:
            # TODO: This is Kepler cadence, fix me.
            self.parm = TrapezoidFitParameters(29.424/60.0/24.0)

        if trp_originalestimates:
            self.origests = trp_originalestimates
        else:
            self.origests = TrapezoidOriginalEstimates()

        if trp_planetestimates:
            self.planetests = trp_planetestimates
        else:
            self.planetests = TrapezoidPlanetEstimates()

        self.physval_names = ['']
        self.fixed = np.array([0])
        self.nparm = 0
        self.physval_mins = np.array([0.0])
        self.physval_maxs = np.array([0.0])
        self.physvals = np.array([0.0])
        self.physvalsavs = np.array([0.0])
        self.bestphysvals = np.array([0.0])
        self.boundedvals = np.array([0.0])
        self.boundedvalsavs = np.array([0.0])
        self.bestboundedvals = np.array([0.0])
        self.model = np.array([0.0])
        self.errscl = 1.0
        self.chi2min = 0.0
        self.minimized = False
        self.sampleit = np.array([0.0])
        self.fitdata = np.array(0, dtype=np.bool)
        self.normlc = np.array([0.0])
        self.normes = np.array([0.0])
        self.normts = np.array([0.0])
        self.normots = np.array([0.0])
        self.timezpt = 0.0

    # TODO: Call from init?
    def trp_setup(self):
        """Setup various data products before minimizing."""
        per = self.origests.period
        eph = self.origests.epoch
        dur = self.origests.duration
        depth = self.origests.depth / 1.0e6
        durday = dur / 24.0
        phidur = dur / 24.0 / per

        # Normalize the time series
        ts = self.normots
        medianEvent = np.median(np.round((ts - eph)/per))
        self.timezpt = eph + (medianEvent * per)
        self.normts = self.normots - self.timezpt
        # identify in transit data to over sample and fitting region
        phi = phase_data(self.normts, per, 0.0)
        self.sampleit = np.where(
            abs(phi) < (phidur * 1.5), self.parm.samplen, 1)
        self.fitdata = np.where(
            abs(phi) < (phidur * self.parm.fitregion), True, False)
        # always fit less than a 0.25 of phase space for stability
        #  and efficiency reasons
        self.fitdata = np.where(abs(phi) > 0.25, False, self.fitdata)

        # Set parameters and bounds
        self.physval_names = ['To', 'Depth', 'BigT', 'TRatio']
        self.physval_mins = np.array([-durday*1.5, 1.0e-6, 0.0, 1.0e-10])
        self.physval_maxs = np.array([durday*1.5, depth*5.0, durday*3.0, 1.0])
        self.fixed = np.array([0, 0, 0, 0])
        self.physvals = np.array([0.0, depth, durday, 0.2])
        self.nparm = np.size(self.fixed)

        # Validate trapezoid fit inputs look reasonable
        self.trp_validate()

        self.modellc = np.full_like(self.normlc, 1.0)
        self.chi2min = self.normlc.size * 2000.0
        self.likecount = 0
        self.set_boundedvals()

        # physvalsavs and boundedvalsavs are used to store parameters
        #  that are fixed during the calculation
        #  ***They must be populated with fixed values before moving forward
        self.physvalsavs = self.physvals
        self.boundedvalsavs = self.boundedvals

        self.bestphysvals = self.physvals
        self.bestboundedvals = self.boundedvals
        self.minimized = False

    def set_boundedvals(self):
        """Convert parameters to bounded versions that the minimizer will use.

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
        """Make a trapezoid model at the given input parameters.  This routine
            generates the ioblk class which is used in the transit model.
            You can save time if you want to generate many models by
            calling this function once to generate the ioblk and then call
            trapezoid_model_raw() to generate the models at other inputs
                bypassing some of the setup routines in this function.
           INPUT:
               ts - Mid cadence time stamps
               period - Period of signal
                        ***assumed fixed during model generation**
               epoch - Estimated epoch of signal.  Must be on same system
                               as ts
               depth [ppm] - Model depth
               big_t [hr] -full transit duration in hours
               little_t [hr] - ingress time in hours
               subsamplen - Subsample each cadence by this factor
            OUTPUT:
                ioblk - structure class containing model ligh curve
                        located at ioblk.modellc
        """
        # Instantiate trp_ioblk class and fill in values
        ioblk = cls()
        ioblk.parm.debugLevel = 0
        ioblk.parm.samplen = subsamplen
        ioblk.normots = ts
        ioblk.origests.period = period
        ioblk.origests.epoch = epoch
        ioblk.origests.depth = depth
        ioblk.origests.duration = big_t
        # Calculate this from timeSeries
        ioblk.parm.cadlen = np.median(np.diff(ts))
        ioblk.trp_setup()
        # update the tratio
        ioblk.physvals[3] = little_t / big_t
        ioblk.set_boundedvals()

        ioblk.physvalsavs = ioblk.physvals
        ioblk.boundedvalsavs = ioblk.boundedvals
        ioblk.trapezoid_model()
        return ioblk

    def trapezoid_model_raw(self, epoch, depth, big_t, little_t):
        """If you have a preexisting ioblk from fit or trapezoid_model_onemodel()
            You can just call this function to get another model
            at a different epoch depth duration and ingress time
            ****period is not variable at this point call
            trapezoid_model_onemodel() instead
           INPUT:
               ioblk - pre-existing ioblk from fitting or
                       trapezoid_model_onemodel()
               epoch - Estimated epoch of signal.  Must be on same system
                               as ts
               depth [ppm] - Model depth
               big_t [hr] -full transit duration in hours
               little_t [hr] - ingress time in hour
            OUTPUT:
                ioblk - structure class containing model ligh curve
                        located at ioblk.modellc
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
        Result is stored in ``self.modellc``.

        Raises
        ------
        ValueError
            Model generation failed.

        """
        to = self.physvals[0]
        depth = self.physvals[1]
        big_t = self.physvals[2]
        tRatio = self.physvals[3]
        little_t = tRatio * big_t
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
            deltaXSmall = cadlen / np.float(samplen)
            smallBlock = np.linspace(-cadlen/2.0 + deltaXSmall/2.0,
                                     cadlen/2.0 - deltaXSmall/2.0, samplen)
            oN = ztmp.size
            ztmp_highres = np.tile(ztmp, samplen)
            ztmp_highres = np.reshape(ztmp_highres, (samplen, oN))
            smallBlock_highres = np.tile(smallBlock, oN)
            smallBlock_highres = np.reshape(smallBlock_highres, (oN, samplen))
            smallBlock_highres = np.transpose(smallBlock_highres)
            ztmp_highres = ztmp_highres + smallBlock_highres
            ztmp_highres = ztmp_highres.ravel(order='F')
            lctmp_highres = trapezoid_vectors(
                ztmp_highres, depth, big_t, little_t)
            nN = ztmp_highres.size
            lctmp = lctmp_highres.reshape([oN, int(nN/oN)]).mean(1)
            lc[idx] = lctmp

        self.modellc = lc
        if np.sum(np.isfinite(lc)) != lc.size:
            raise ValueError('Trapezoid model failed')

    def trp_validate(self):
        # Check that physvals are within limits
        if (np.any(np.greater_equal(self.physvals, self.physval_maxs))):
            raise ValueError(f'physvals: {self.physvals} is greater than '
                             f'physval_maxs: {self.physval_maxs}')
        if (np.any(np.less_equal(self.physvals, self.physval_mins))):
            raise ValueError(f'physvals: {self.physvals} is less than '
                             f'physval_mins: {self.physval_mins}')
        # Check for NaNs in input data series
        if (np.any(np.isnan(self.normlc))):
            raise ValueError("TrapFit: Input light curve contains NaN")
        if (np.any(np.isnan(self.normes))):
            raise ValueError(
                "TrapFit: Input uncertainty estimate contains NaN")
        if (np.any(np.isnan(self.normots))):
            raise ValueError("TrapFit: Input time data contains NaN")
        # Check for input data series that has negative flux data should be
        #  normalized to 1.0
        if (np.any(np.less(self.normlc, 0.0))):
            raise ValueError("TrapFit: Negative Flux in light curve")

    def trp_likehood(self, pars):
        """Return a residual time series of data minus model
           trp_setup(ioblk) should be called before this function is called
           INPUT:
           pars - [numpy array] vector of parameter values
           ioblk - [class] trp_ioblk class structure
           OUTPUT:
           residuals - sum of squares of residuals of data - model
           ioblk - [class] modified ioblk
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
        residuals = (self.normlc[idx] -
                     self.modellc[idx]) / (self.normes[idx] * self.errscl)
        # Return scalar summed residuals
        residuals = np.sum(residuals**2)

        return residuals

    # TODO: Not sure what this was for but plotting should be optional,
    # so this was taken out of the function being passed into scipy optimize.
    # Fix as needed.
    def plot_likehood(self):
        """Plot results from :meth:`trp_likehood`."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(3, 2), dpi=300, facecolor='white')
        ax = fig.subplots()
        ax.set_position([0.125, 0.125, 0.825, 0.825])
        ax.set_facecolor('white')

        period = self.origests.period
        tzero = self.physvals[0]
        ts = self.normts
        phi = phase_data(ts, period, tzero)

        ax.plot(phi, self.normlc, '.', markersize=0.6)
        ax.plot(phi, self.modellc, '.r', markersize=0.6)
        plt.draw()

    def trp_iterate_solution(self, nIter):
        """Peform multiple iterations starting from random initial conditions
           return the best solution in a chi2 sense among the nIter iterations
        """
        import scipy.optimize as opt

        bestChi2s = np.zeros(nIter)
        bestParameters = np.zeros((self.physvals.size, nIter))
        gdFits = np.zeros(nIter, dtype=np.bool)
        depth = self.origests.depth / 1.0e6
        for i in range(nIter):
            self.physvals = self.physval_mins + \
                            np.random.rand(self.physvals.size) * \
                            (self.physval_maxs - self.physval_mins)
            # Force depth parameter to start at minimum half the depth
            if self.physvals[1] < np.abs(depth/2.0):
                self.physvals[1] = depth / 2.0
            # Replace random starts with parameters values that are fixed
            self.physvals = np.where(self.fixed == 1, self.physvalsavs,
                                     self.physvals)
            self.set_boundedvals()
            freeidx = np.where(self.fixed == 0)[0]
            startParameters = self.boundedvals[freeidx]
            # usemethod = 'Nelder-Mead'
            usemethod = 'Powell'
            useoptions = {'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 2000,
                          'maxfev': 2000}
            # usemethod = 'CG'
            # useoptions = {'gtol': 1e-5, 'maxiter': 2000}

            # TODO: Need to see how opt works
            allOutput = opt.minimize(self.trp_likehood, startParameters,
                                     method=usemethod, options=useoptions)
            self.boundedvals[freeidx] = allOutput['x']
            self.boundedvals = np.where(self.fixed == 1, self.boundedvalsavs,
                                        self.boundedvals)
            self.set_physvals()
            chi2min = allOutput['fun']
            self.logger.debug(f"It: {i}  Chi2: {chi2min}")
            self.logger.debug(f"{self.physvals}")
            if np.isfinite(self.physvals).all():
                gdFits[i] = True
                bestChi2s[i] = chi2min
                bestParameters[:, i] = self.physvals

        # Done with iterations find the best one by chi2min
        bestMaskedIdx = np.argmin(bestChi2s[gdFits])
        self.chi2min = bestChi2s[gdFits][bestMaskedIdx]
        self.bestphysvals = bestParameters[:, gdFits][:, bestMaskedIdx]
        self.physvals = self.bestphysvals
        self.set_boundedvals()
        self.bestboundedvals = self.boundedvals
        self.logger.debug(f"Overall Best Chi2 Min: {self.chi2min}")
        self.logger.debug(f"{self.physvals}")
        self.minimized = True

    def trp_estimate_planet(self):
        """Convert the trapezoid fit solution into a crude estimate
           of a planet model that is close to trapezoid solution
           This fills out values in trp_planetestimates class
        """
        if not self.minimized:
            raise ValueError("Getting planet estimates for non-converged"
                             "trapezoid fit is not allowed")
        self.planetests.period = self.origests.period
        self.planetests.epoch = self.timezpt + self.bestphysvals[0]
        self.planetests.big_t = self.bestphysvals[2]
        self.planetests.little_t = self.bestphysvals[3] * self.planetests.big_t
        self.planetests.depth = self.bestphysvals[1]
        # call likehood to get best transit model
        idx = np.where(self.fixed == 0)[0]
        self.trp_likehood(self.bestboundedvals[idx])
        trapmodlc = self.modellc
        self.planetests.minDepth = (1.0 - trapmodlc.min()) * 1.0e6
        self.planetests.radiusRatio = np.sqrt(self.planetests.minDepth / 1.0e6)
        self.planetests.impactParameter = np.sqrt(
            1.0 - np.amin([self.planetests.radiusRatio *
                           self.planetests.big_t / self.planetests.little_t,
                           1.0]))
        self.planetests.tauzero = np.sqrt(
            self.planetests.big_t * self.planetests.little_t / 4.0 /
            self.planetests.radiusRatio)
        self.planetests.semiaxisRatio = (
            self.planetests.period / 2.0 / np.pi / self.planetests.tauzero)
        mu = np.sqrt(1.0 - self.planetests.impactParameter**2)
        self.planetests.surfaceBright = (
            1.0 - self.planetests.u1*(1.0-mu) - self.planetests.u2*(1.0-mu)**2)
        self.planetests.equivRadiusRatio = (
            self.planetests.radiusRatio /
            np.sqrt(self.planetests.surfaceBright))


# TODO: Surely this can be a method somehow?!
def trapezoid_fit(timeSeries, dataSeries, errorSeries,
                  signalPeriod, signalEpoch, signalDuration, signalDepth,
                  fitTrialN=13, fitRegion=4.0, errorScale=1.0, debugLevel=0,
                  sampleN=15, showFitInterval=30):
    """Perform a trapezoid fit to a normalized flux time series
       Assumes all data has the same cadence duration
       Period is fixed during the trapezoid fit

       INPUT:
           timeSeries - Mid cadence time stamps
           dataSeries - Normalized time series
           errorSeries - Error time series
           signalPeriod - Period of signal ***assumed fixed during model fit**
           signalEpoch - Estimated epoch of signal.  Must be on same system
                           as timeSeries
           signalDuration [hr] - Estimated signal duration ***In hours**
           signalDepth [ppm] - Estimated signal depth
           fitTrialN - How many trial fits to perform starting at random
                       initial locations.  Increase this if you find the
                       minimization is returning local minima
           fitRegion - Fit data within fitRegion*signalDuration of signalEpoch
           errorScale - Default 1.0 - Scale the errorbars by this factor
           debugLevel - 0 Show nothing; 1-Show some text about iterations
                        2 Show some more text; 3 - Show graphical fit in
                           progress; 4 - pause for each graphical fit
           sampleN - Subsample each cadence by this factor
           showFitInterval - If debugLevel >=3 the show every showFitInterval
                              function evaluation
        OUTPUT:
           ioblk - An instance of trp_ioblk which is a class used to store
                   all information pertaining to fit results
    """
    # Instantiate trp_ioblk class and fill in values
    ioblk = TrapezoidFit()
    ioblk.parm.debugLevel = debugLevel
    ioblk.parm.samplen = sampleN
    ioblk.parm.likehoodmoddisplay = showFitInterval
    ioblk.fitregion = fitRegion
    ioblk.normlc = dataSeries
    ioblk.normes = errorSeries
    ioblk.errscl = errorScale
    ioblk.normots = timeSeries
    ioblk.origests.period = signalPeriod
    ioblk.origests.epoch = signalEpoch
    ioblk.origests.duration = signalDuration  # input duration is hours
    ioblk.origests.depth = signalDepth

    # Calculate this from timeSeries
    ioblk.parm.cadlen = np.median(np.diff(timeSeries))

    # setup some more variables
    ioblk.trp_setup()
    # Find solution by trying random initial conditions
    ioblk.trp_iterate_solution(fitTrialN)
    # Convert the trapezoid fit solution into a pseudo planet model parameters
    ioblk.trp_estimate_planet()

    # Raise an exception if final model is consistent with flat
    if (np.all(np.abs(ioblk.modellc - ioblk.modellc[0])
               < (10.0 * sys.float_info.epsilon))):
        raise ValueError("TrapFit: Model light curve is flat!")
    # Check for NaNs in output model
    if (np.any(np.isnan(ioblk.modellc))):
        raise ValueError("TrapFit: Output Model light curve contains NaN")

    return ioblk
