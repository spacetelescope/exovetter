"""Utility functions."""

import sys
import warnings
from exovetter import lightkurve_utils
from exovetter import utils
from exovetter import const as exo_const
import astropy.units as u
from exovetter import vetters as vet
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os

__all__ = ['sine', 'estimate_scatter', 'mark_transit_cadences', 'median_detrend', 
           'plateau', 'set_median_flux_to_zero', 'set_median_flux_to_one', 'sigmaClip', 
           'get_mast_tce', 'WqedLSF', 'compute_phases', 'first_epoch', 'run_all']

def sine(x, order, period=1):
    """Sine function for SWEET vetter."""
    w = 2 * np.pi / period
    if order == 0:
        return np.sin(w * x)
    elif order == 1:
        return np.cos(w * x)
    else:
        raise ValueError("Order should be zero or one")


def estimate_scatter(flux):
    """Estimate the typical scatter in a lightcurve.

    Uses the same method as Marshall vetter (Mullally et al 2017)

    Parameters
    ----------
    flux
        (np 1d array). Flux to measure scatter of. Need not have
        zero mean.

    Returns
    -------
    result : float
        Scatter of data in the same units as in the input ``flux``


    Notes
    -----
    Algorithm is reasonably insensitive to outliers. For best results
    uses outlier rejection on your lightcurve before computing scatter.
    Nan's and infs in lightcurve will propegate to the return value.
    """

    flux = flux[np.isfinite(flux)]
    diff = np.diff(flux)

    # Remove egregious outliers. Shouldn't make much difference
    idx = sigmaClip(diff, 5)
    diff = diff[~idx]

    # TODO: should this be a running mean?
    mean = np.mean(diff)
    mad = np.median(np.fabs(diff - mean))
    std = 1.4826 * mad

    # std is the rms of the diff. std on single point
    # is 1/sqrt(2) of that value,
    return float(std / np.sqrt(2))


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

    idx = np.zeros_like(time, dtype=np.bool_) 
    # Changed from np.bool8 to np.bool_ for py 3.11  
    for tt in transit_times:
        diff = time - tt
        diff[flags] = big_num
        if not np.all(np.isfinite(diff)):
            raise ValueError('NaN found in diff of transit time')

        idx = np.bitwise_or(idx, np.fabs(diff) < max_diff)

    if not np.any(idx):
        warnings.warn('No cadences found matching transit locations')

    return idx


def median_detrend(flux, nPoints):
    """Quick and dirty median smooth function.
    Not designed to be at all efficient.
    """
    size = len(flux)
    filtered = np.zeros_like(flux)
    for i in range(size):
        lwr = max(i - nPoints, 0)
        upr = min(lwr + 2 * nPoints, size)
        lwr = upr - 2 * nPoints

        sub = flux[lwr:upr]

        offset = np.median(sub)
        filtered[i] = flux[i] - offset

    return filtered


def plateau(array, threshold):
    """Find plateaus in an array, i.e continuous regions that exceed threshold.

    Given an array of numbers, return a 2d array such that
    out[:,0] marks the indices where the array crosses threshold from
    below, and out[:,1] marks the next time the array crosses that
    same threshold from below.

    Parameters
    ----------
    array : array_like
        Numpy 1D array 

    threshold : float or array_like
        If threshold is a single number, any point 
        above that value is above threshold. If it's an array,
        it must have the same length as the first argument, and
        an array[i] > threshold[i] to be included as a plateau
    
    Returns
    -------
    loc : array_like
        Numpy 2d array with 2 columns.


    Notes
    -----
    To find the length of the plateaus, use
    out[:,1] - out[:,0]

    To find the length of the largest plateau, use
    np.max(out[:,1] - out[:,0])

    The algorithm fails if a value is exactly equal to the threshold.
    To guard against this, we add a very small amount to threshold
    to ensure floating point arithmetic prevents two numbers being
    exactly equal.
    """
    
    arr = array.astype(np.float32)
    arr = arr - threshold + 1e-12
    arrPlus = np.roll(arr, 1)

    # Location of changes from -ve to +ve (or vice versa)
    # Last point is bogus , so we calcualte it by hand
    sgnChange = arr * arrPlus

    # Roll around can't compute sign change for zeroth elt.
    sgnChange[0] = +1
    if arr[0] > 0:
        sgnChange[0] = -1

    loc = np.where(sgnChange < 0)[0]

    if np.fmod(len(loc), 2) != 0:
        locc = loc.copy()
        locc.resize(len(loc) + 1, refcheck=False)
        loc = locc.copy()
        loc[-1] = len(arr)

    if len(loc) == 0:
        return []
    return loc.reshape((-1, 2))


def set_median_flux_to_zero(flux):
    """Set median flux to zero."""
    flux = set_median_flux_to_one(flux)
    return flux - 1


def set_median_flux_to_one(flux):
    """Set median flux to one."""
    if not np.all(np.isfinite(flux)):
        raise ValueError('flux must contain all finite values')

    medflux = np.median(flux)
    if np.isclose(medflux, 0):
        return flux + 1

    flux = flux.copy()
    flux /= medflux

    return flux


def sigmaClip(y, nSigma, maxIter=1e4, initialClip=None):
    """Iteratively find and remove outliers

    Find outliers by identifiny all points more than **nSigma** from
    the mean value. The recalculate the mean and std and repeat until
    no more outliers found.

    Parameters
    ----------
    y : ndarray
        Array to be cleaned
    nSigma : float
        Threshold to cut at.
        5 is typically a good value for
        most arrays found in practice.
    maxIter : int, optional
        Maximum number of iterations
    initialClip : ndarray, optional
        1D boolean array.
        If an element of initialClip is set to True,
        that value is treated as a bad value in the first iteration, and
        not included in the computation of the mean and std.

    Returns
    -------
    idx : ndarray
        1d numpy array. Where set to True, the corresponding element of y
        is an outlier.
    """

    idx = initialClip
    if initialClip is None:
        idx = np.zeros(len(y), dtype=bool)

    if (len(idx) != len(y)):
        raise AssertionError('length of array y not equal to initialClip')

    # x = np.arange(len(y))
    # mp.plot(x, y, 'k.')

    oldNumClipped = np.sum(idx)
    for i in range(int(maxIter)):
        mean = np.nanmean(y[~idx])
        std = np.nanstd(y[~idx])

        newIdx = np.fabs(y - mean) > nSigma * std
        newIdx = np.logical_or(idx, newIdx)
        newNumClipped = np.sum(newIdx)

        # print "Iter %i: %i (%i) clipped points " \
        # %(i, newNumClipped, oldNumClipped)

        if newNumClipped == oldNumClipped:
            return newIdx

        oldNumClipped = newNumClipped
        idx = newIdx
        i += 1
    return idx


def get_mast_tce(name, catalogs=None):
    """
    Populate a TCE object using the mast database

    Parameters
    ----------
    name : str
        Star name, TCE, or TOI
    
    catalog : list of str
        Name of catalog(s) to use

    Returns
    -------
    tce : tce.Tce object
     populates the object with period, epoch, epoch_offset, depth, duration

    """
    
    import requests
    from exovetter.tce import Tce
    import astropy.units as u
    from exovetter import const
    import numpy as np
    
    planeturl = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
    header = {}
    
    url = planeturl + name + "/properties/"

    r = requests.get(url = url, headers = header)
    
    if len(r.json()) < 1:
        print("No TCE Information was returned from MAST.")
        return []
        
    tces = []
    
    for prop in r.json():
        try:
            if catalogs is not None and np.invert(any(ele in prop['catalog_name'] for ele in catalogs)): # if current prop catalog isnt in input list don't add it
                continue
            period = prop['orbital_period']
            punit = prop['orbital_period_unit']
            epoch = prop['transit_time']
            #epoch_offset_str = 'mjd'
            epoch_offset_str = prop['transit_time_unit'].lower()
            depth = prop['transit_depth']
            duration = prop['transit_duration']
            catalog_name = prop['catalog_name']
            if duration is None:
                duration  = 0
            durunit = prop['transit_duration_unit']
            if durunit is None:
                durunit = "d"
            
            atce = Tce(period = period * u.__dict__[punit],
                       epoch = epoch * u.d,
                       epoch_offset = const.__dict__['string_to_offset'][epoch_offset_str],
                       depth = depth * const.frac_amp,
                       duration = duration * u.__dict__[durunit],
                       target = name,
                       catalog_name = catalog_name
                       )
            
            tces.append(atce)
        except SyntaxError:
            print("Incorrect syntax")
            prop
            pass

    if catalogs is not None and len(tces) < 1:
        print('no tces from input catalog(s) found')     
    return tces

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

        A = np.dot(df.T, df)
        covar = np.linalg.inv(A)

        wy = self.y / self.s
        df = df.T # transposed to work with np.dot updates MD 2023
        beta = np.dot(df, wy) # swapped order in dot product MD 2023
        params = np.dot(beta, covar)

        # Store results
        self._param = params
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


def compute_phases(time, period, epoch, offset=0.5):
    """Calculate phases.

    Parameters
    ----------
    time : float
        Time

    period : float
        Period in units of time.

    epoch : float
        Time of the transit in units of the time.

    offset : float
        Fractional value the times of the epoch should land on.

    Returns
    -------
    phases : float
        Phases in units of the period. All phases are positive in value.

    """
    # Determine Epoch just less than the smallest time before folding
    n_periods = np.abs(np.min(time) - epoch) / period
    if epoch > np.min(time):
        epoch0 = epoch - np.ceil(n_periods) * period
    else:
        epoch0 = epoch + np.floor(n_periods) * period
    phases = np.fmod(time - epoch0 + (offset * period), period)
    pmin = np.min(phases)

    return phases

def first_epoch(epoch, period, lc):
    """Returns epoch of the first time in a lc file. Usefull when pulling down TCEs from MAST.

    Parameters
    ----------
    epoch : float
        Time of transit from a given TCE in same time system as lc.

    period : float
        Period in units of time.

    lc : lightkurve object
        lightkurve object with flux and time.

    Returns
    -------
    first_epoch : float
        Epoch of first transit in lc in the time system of the lc. 

    """
    # TODO might need to assert epoch.value is within lc.time.value
    if epoch.value >= lc.time.value[0]:
        N = np.floor((epoch.value-lc.time.value[0])/period.value)
        first_epoch = epoch - N*period
    else:
        N = np.ceil((lc.time.value[0]-epoch.value)/period.value)
        first_epoch = epoch + N*period

    return first_epoch

def run_all(tce, lc, tpf=None, vetters=None, remove_metrics=None, plot=False, plot_dir=None):
    """Run a set of vetters on a tce and lc, returning a dictionary of all metrics collected
    
    Parameters
    ----------
    tce : tce object
        tce object is a dictionary that contains information about the tce
        to vet, like period, epoch, duration, depth

    lc : lightkurve object
        lightkurve object with the time and flux to use for vetting.

    tpf : obj
        ``lightkurve`` target pixel file object with pixels in column lc_name

    vetters : list
        list of vetter objects to run on, ie [vet.ModShift(), vet.OddEven(dur_frac=0.75)]
        Defaults to all vetters
    
    remove_metrics : list
        metrics to not store, defaults to removing plotting values

    plot : bool
        Option to return a pdf of the vetting diagnostic plots, defaults to False

    plot_dir : str
        Path to store diagnostic pdfs in, defaults to current working directory

    Returns
    -------
    results_dict : dictionary
        Dictionary of the kept vetting metrics
    """
    
    # Set initial parameters
    if not vetters:
        vetters = [vet.VizTransits(), vet.ModShift(), vet.Lpp(), vet.OddEven(), vet.TransitPhaseCoverage(), vet.Sweet(), vet.LeoTransitEvents(), vet.Centroid()]

    if not remove_metrics:
        remove_metrics = ['plot_data']

    if not plot_dir:
        plot_dir = os.getcwd()+'/'

    if not tpf:
        if any(vetter.__class__.__name__ == 'Centroid' for vetter in vetters):
            raise Exception("TPF file required while running centroid")

    # Run all listed vetters
    results_list = []

    if not plot:
        for vetter in vetters:
            if vetter.__class__.__name__ != 'Centroid':
                vetter_results = vetter.run(tce, lc, plot=False) # dictionary returned from each vetter
                results_list.append(vetter_results)
            else:
                vetter_results = vetter.run(tce, tpf, plot=False) # centroid uses tpf rather than lc
                results_list.append(vetter_results)

    else:
        plot_name = lc.LABEL

        diagnostic_plot = PdfPages(plot_dir+plot_name+'.pdf') # initialize a pdf to save each figure into
        plot_figures = []
        
        # Manually run viz transits with an extra mark cadences plot
        cadences_plot = mark_cadences_plot(lc, tce)
        diagnostic_plot.savefig(cadences_plot)

        for vetter in vetters:
            if vetter.__class__.__name__ not in ['VizTransits', 'Centroid', 'LeoTransitEvents']: # viz_transits and Centroid generate more than one figures so handle later
                vetter_results = vetter.run(tce, lc, plot=True) # dictionary returned from each vetter
                plot_figures.append(plt.gcf())
                plt.close() # Make them not show up if running in a notebook
                results_list.append(vetter_results)

            if vetter.__class__.__name__ == 'Centroid': # centroid produces 2 plots, the second of which is the most useful so just collect that one
                vetter_results = vet.Centroid(lc_name=vetter.lc_name, diff_plots=False, centroid_plots=True).run(tce, tpf)
                plot_figures.append(plt.gcf())
                plt.close()
                results_list.append(vetter_results)
                
        # run viz_transits plots
        transit = vet.VizTransits(transit_plot=True, folded_plot=False).run(tce, lc)
        transit_plot = plt.gcf()
        transit_plot.suptitle(plot_name+' Transits')
        transit_plot.tight_layout()
        plt.close()
        diagnostic_plot.savefig(transit_plot)

        folded = vet.VizTransits(transit_plot=False, folded_plot=True).run(tce, lc)
        folded_plot = plt.gcf()
        folded_plot.suptitle(plot_name+' Folded Transits')
        folded_plot.tight_layout()
        plt.close()
        diagnostic_plot.savefig(folded_plot)
        
        # Save each diagnostic plot stored in plot_figures to diagnostic_plot pdf file
        for plot in plot_figures:
            diagnostic_plot.savefig(plot)

        diagnostic_plot.close()
    
    # Convert to a single dictionary output
    results_dict = {k: v for d in results_list for k, v in d.items()} # Combine all dictionaries returned from vetters

    # delete specified metrics
    for key in remove_metrics:
        if results_dict.get(key):
            del results_dict[key]
    
    return results_dict


def mark_cadences_plot(lc, tce):
    """return figure object of the lightcurve with epochs oeverplotted"""
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9,5))

    # Get lightcurve data
    time, flux, time_offset_str = lightkurve_utils.unpack_lk_version(lc, "flux")  # noqa: E50
    time_offset_q = getattr(exo_const, time_offset_str)
    epoch = tce.get_epoch(time_offset_q).to_value(u.day)
    
    # Get points of lightcurve found to be in transit
    period = tce["period"].to_value(u.day)
    dur = tce["duration"].to_value(u.day)
    intransit = mark_transit_cadences(time, period, epoch, dur)

    # Plot epoch
    ax1.axvline(x=epoch, lw='0.6', color='r', label='epoch', alpha=0.5)

    # Plot transit train
    ax1.plot(time, flux, lw=0.72, alpha=0.9)
    ax1.scatter(time, flux, color='k', s=3, label='cadences', alpha=0.5)
    
    # TODO This only plots forward in time from the epoch, works fine assuming tce epoch is first in light curve but not robust
    transit_epochs = epoch-dur/2
    while transit_epochs <= time[-1]:
        ax1.axvline(x=transit_epochs, lw='0.6', color='r', alpha=0.3, ls='--')
        transit_epochs = transit_epochs+dur
        ax1.axvline(x=transit_epochs, lw='0.6', color='r', alpha=0.3, ls='--')
        transit_epochs = transit_epochs-dur + period
        
    # Plot cadences in transit
    ax1.scatter(time[intransit], flux[intransit], color='r', s=4, label='cadences in transit');

    # Plotting params
    ax1.set_ylabel('Flux')
    ax1.set_xlabel('Time '+time_offset_str)
    ax1.set_title(lc.label+' period='+'{0:.2f}'.format(period)+'d, dur='+'{0:.2f}'.format(dur)+'d')
    ax1.legend();

    cadences_plot = plt.gcf()
    plt.close()

    return cadences_plot

    
