"""Module to handle exoplanet vetters."""

from abc import ABC, abstractmethod

from exovetter import lpp

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
        target = self.tce['target_name']
        lpp.plot_lpp_diagnostic(self.plot_data, target, self.norm_lpp)


# TODO: Implement me!
# NOTE: We can have many such tests.
class OddEven(BaseVetter):
    """Odd-even test."""

    # Actual implementation of LPP is called here
    pass
