"""Module to handle exoplanet vetters."""

import os
import warnings
from abc import ABC, abstractmethod

import numpy as np

from exovetter import lpp
from exovetter.utils import mark_transit_cadences, WqedLSF

__all__ = ['BaseVetter', 'Lpp', 'Sweet']


class BaseVetter(ABC):
    """Base class for vetters.
    Each vetting test should be a subclass of this class.

    Parameters
    ----------
    kwargs : dict
        Store the configuration parameters common to all
        Threshold Crossing Events (TCEs).
        For example, for the Odd-even test, it might specify the significance
        of the depth difference that causes a TCE to fail.

    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, tce, lightcurve):
        """Run the vetter on the specified Threshold Crossing Event (TCE)
        and lightcurve to obtain metric.

        Parameters
        ----------
        tce : `~exovetter.tce.TCE`
            TCE.

        lightcurve : obj
            ``lightkurve`` object that contains the detrended lightcurve's
            time and flux arrays.

        Returns
        -------
        result : dict
            A dictionary of metric values.

        """
        pass

    @abstractmethod
    def plot(self):
        """Generate a diagnostic plot."""
        pass


class Lpp(BaseVetter):
    """Class to handle LPP Vetter functionality.

    Parameters
    ----------
    map_filename : str or `None`
        Full path to a LPP ``.mat`` file.
        See `~exovetter.lpp.Loadmap`.

    lc_name : str
        Name of the flux array in the ``lightkurve`` object.

    Attributes
    ----------
    map_info : `~exovetter.lpp.Loadmap`
        Map info from ``map_filename``.

    lc_name : str
        Input ``lc_name``.

    tce, lc
        Inputs to :meth:`run`. TCE for this vetter should also
        contain ``snr`` estimate.

    lpp_data : `exovetter.lpp.Lppdata`
        Populated by :meth:`run`.

    raw_lpp : float
        Raw LPP value, populated by :meth:`run`.

    norm_lpp : float
        LPP value normalized by period and SNR, populated by :meth:`run`.

    plot_data : dict
        The folded, binned transit prior to the LPP transformation,
        populated by :meth:`run`.

    """
    def __init__(self, map_filename=None, lc_name="flux"):
        self.map_info = lpp.Loadmap(filename=map_filename)
        self.lc_name = lc_name
        self.tce = None
        self.lc = None
        self.norm_lpp = None
        self.raw_lpp = None
        self.plot_data = None

    def run(self, tce, lightcurve):
        self.tce = tce
        self.lc = lightcurve
        self.lpp_data = lpp.Lppdata(self.tce, self.lc, self.lc_name)

        self.norm_lpp, self.raw_lpp, self.plot_data = lpp.compute_lpp_Transitmetric(self.lpp_data, self.map_info)  # noqa: E501

        # TODO: Do we really need to return anything if everything is stored as
        # instance attributes anyway?
        return {
            'raw_lpp': self.raw_lpp,
            'norm_lpp': self.norm_lpp,
            'plot_data': self.plot_data}

    def plot(self):
        if self.plot_data is None:
            raise ValueError(
                'LPP plot data is empty. Execute self.run(...) first.')

        # target is populated in TCE, assume it already exists.
        target = self.tce.target_name
        lpp.plot_lpp_diagnostic(self.plot_data, target, self.norm_lpp)


class Sweet(BaseVetter):
    """Class to handle SWEET Vetter functionality.

    Parameters
    ----------
    threshold_sigma : float
        Threshold for comparing signal to transit period.

    Attributes
    ----------
    tce, lc
        Inputs to :meth:`run`.

    result : dict
        ``'amp'`` contains the best fit amplitude, its uncertainty, and
        amplitude-to-uncertainty ratio for half-period, period, and
        twice the period. ``'msg'`` contains warnings, if applicable.
        Populated by :meth:`run`.

    lsf : `~exovetter.utils.WqedLSF`
        Least squares fit object, populated by :meth:`run`.

    """
    def __init__(self, threshold_sigma=3):
        self.tce = None
        self.lc = None
        self.result = None
        self.lsf = None
        self.threshold_sigma = threshold_sigma

    def run(self, tce, lightcurve):
        self.tce = tce
        self.lc = lightcurve
        self.result, self.lsf = self._do_fit(
            lightcurve.time, lightcurve.flux,
            tce.period, tce.tzero, tce.duration)

    def plot(self):
        import matplotlib.pyplot as plt

        phase = self.lsf.x
        flux = self.lsf.y
        best_fit = self.lsf.get_best_fit_model()

        fig, ax = plt.subplots()
        ax.plot(phase, flux, 'k.')
        ax.plot(phase, best_fit, 'r.')
        ax.set_title(f'{self.tce.target_name} ({self.tce.event_name})')
        plt.draw()

        return fig

    def _do_fit(self, time, flux, period_days, epoch, duration_hrs):

        if len(time) != len(flux):
            raise ValueError('time and flux length mismatch')

        threshold_sigma = self.threshold_sigma

        idx = np.isnan(time) | np.isnan(flux)
        time = time[~idx]
        flux = flux[~idx]

        duration_days = duration_hrs / 24.
        idx = mark_transit_cadences(time, period_days, epoch, duration_days)
        flux = flux[~idx]

        out = []
        for per in [period_days * 0.5, period_days, 2 * period_days]:
            phase = np.fmod(time - epoch + per, per)
            phase = phase[~idx]
            period = np.max(phase)
            f_obj = WqedLSF(phase, flux, None, period=period)
            amp, amp_unc = f_obj.compute_amplitude()
            out.append([amp, amp_unc, amp / amp_unc])
        result = np.array(out)

        msg = []
        if result[0, -1] > threshold_sigma:
            warn_text = 'SWEET test finds signal at HALF transit period'
            msg.append(f"WARN: {warn_text}")
            warnings.warn(warn_text)
        if result[1, -1] > threshold_sigma:
            warn_text = "SWEET test finds signal at the transit period"
            msg.append(f"WARN: {warn_text}")
            warnings.warn(warn_text)
        if result[2, -1] > threshold_sigma:
            warn_text = "SWEET test finds signal at TWICE the transit period"
            msg.append(f"WARN: {warn_text}")
            warnings.warn(warn_text)
        if len(msg) == 0:
            msg = [("OK: SWEET finds no out-of-transit variability at "
                    "transit period")]

        return {'msg': os.linesep.join(msg), 'amp': result}, f_obj


# TODO: Implement me!
# NOTE: We can have many such tests.
class OddEven(BaseVetter):
    """Odd-even test."""

    # Actual implementation of LPP is called here
    pass
