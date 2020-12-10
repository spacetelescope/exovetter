"""Module to handle exoplanet vetters."""

from abc import ABC, abstractmethod

from exovetter import lpp
from exovetter import odd_even
from exovetter import transit_coverage

import astropy.units as u
import exovetter.const as exo_const

__all__ = ['BaseVetter', 'Lpp']


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
        """Run the vetting test.

        Parameters
        ----------
        tce : `~exovetter.tce.TCE`
            TCE.

        lightcurve : array_like
            Lightcurve data.

        Returns
        -------
        result : dict
            A dictionary of metric values.

        """
        pass

    @abstractmethod
    def plot(self, tce, lightcurve):
        """Generate a diagnostic plot.

        Parameters
        ----------
        tce, lightcurve
            See :meth:`run`.

        """
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
        Inputs to :meth:`run`.

    lpp_data, raw_lpp, norm_lpp, plot_data
        Results populated by :meth:`run`.

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
        """Run the LPP Vetter on the specified Threshold Crossing Event (TCE)
        and lightcurve to obtain metric.

        Parameters
        ----------
        tce : `~exovetter.tce.Tce`
            In addition to required quantities, it should also contain
            ``snr`` estimate.

        lightcurve : obj
            ``lightkurve`` object that contains the detrended lightcurve's
            time and flux arrays.

        Returns
        --------
        result : dict
            A dictionary of metric:

            * ``raw_lpp`` (float): Raw LPP value.
            * ``norm_lpp`` (float): LPP value normalized by period and SNR.
            * ``plot_data`` (dict): The folded, binned transit prior to the
              LPP transformation.

        """
        self.tce = tce
        self.lc = lightcurve

        self.lpp_data = lpp.Lppdata(self.tce, self.lc, self.lc_name)

        self.norm_lpp, self.raw_lpp, self.plot_data = \
            lpp.compute_lpp_Transitmetric(self.lpp_data, self.map_info)

        # TODO: Do we really need to return anything if everything is stored as
        # instance attributes anyway?
        return {
            'raw_lpp': self.raw_lpp,
            'norm_lpp': self.norm_lpp,
            'plot_data': self.plot_data}

    def plot(self):  # pragma: no cover
        if self.plot_data is None:
            raise ValueError(
                'LPP plot data is empty. Execute self.run(...) first.')

        # target is populated in TCE, assume it already exists.
        target = self.tce['target_name']
        lpp.plot_lpp_diagnostic(self.plot_data, target, self.norm_lpp)


class OddEven(BaseVetter):
    """Odd-even Metric"""

    def __init__(self, lc_name="flux"):
        self.lc_name = lc_name
        self.odd_depth = None
        self.even_depth = None
        self.sigma = None

    def run(self, tce, lightcurve, dur_frac=0.3):

        self.time = lightcurve.time
        self.flux = getattr(lightcurve, self.lc_name)
        self.dur_frac = dur_frac
        time_offset_str = lightcurve.time_format
        time_offset_q = getattr(exo_const, time_offset_str)

        self.period = tce['period'].to_value(u.day)
        self.duration = tce['duration'].to_value(u.day)
        self.epoch = tce.get_epoch(time_offset_q).to_value(u.day)

        self.sigma, self.odd_depth, self.even_depth = \
            odd_even.calc_odd_even(self.time, self.flux, self.period,
                                   self.epoch, self.duration, ingress=None,
                                   dur_frac=self.dur_frac)

    def plot(self):  # pragma: no cover

        odd_even.diagnostic_plot(self.time, self.flux, self.period,
                                 self.epoch, self.duration*self.dur_frac,
                                 self.odd_depth, self.even_depth)


class TransitPhaseCoverage(BaseVetter):
    """Transit Phase Coverage"""

    def __init__(self, lc_name="flux"):
        self.lc_name = lc_name

    def run(self, tce, lightcurve, nbins=10, ndur=2):

        time = lightcurve.time
        self.time = time
        self.flux = getattr(lightcurve, self.lc_name)

        p_day = tce['period'].to_value(u.day)
        dur_hour = tce['duration'].to_value(u.hour)
        time_offset_str = lightcurve.time_format
        time_offset_q = getattr(exo_const, time_offset_str)
        epoch = tce.get_epoch(time_offset_q).to_value(u.day)

        self.tp_cover, self.hist, self.bins = \
            transit_coverage.calc_coverage(time, p_day, epoch, dur_hour,
                                           ndur=ndur, nbins=nbins)

    def plot(self):  # pragma: no cover

        transit_coverage.plot_coverage(self.hist, self.bins)
