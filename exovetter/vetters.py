"""Module to handle exoplanet vetters."""

import pprint
from abc import ABC, abstractmethod

import astropy.units as u

from exovetter.centroid import centroid as cent
from exovetter import transit_coverage
from exovetter import modshift
from exovetter import odd_even
from exovetter import sweet
from exovetter import lpp
from exovetter import const as exo_const
from exovetter import lightkurve_utils
from exovetter import utils
from exovetter import const
from exovetter import model
from exovetter import viz_transits

class BaseVetter(ABC):
    """Base class for vetters.

    Each vetting test should be a subclass of this class.

    Parameters
    ----------
    kwargs
        (dict) Store the configuration parameters common to all
        Threshold Crossing Events (TCEs).
        For example, for the Odd-even test, it might specify the significance
        of the depth difference that causes a TCE to fail.

    """

    def __init__(self, **kwargs):
        self.metrics = None

    def name(self):
        name = str(type(self)).split(".")[-1][:-2]
        return name

    def __str__(self):
        try:
            if self.metrics is None:
                return "{}"  # An empty dictionary
        except AttributeError:
            # No metrics attribute, fall back on repr
            return self.__repr__()

        return pprint.pformat(self.metrics)

    @abstractmethod
    def run(self, tce, lightcurve):
        """Run the vetter on the specified Threshold Crossing Event (TCE)
        and lightcurve to obtain metric.

        Parameters
        ----------
        tce : `~exovetter.tce.Tce`
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

    def plot(self, tce, lightcurve):
        """Generate a diagnostic plot.

        Parameters
        ----------
        tce, lightcurve
            See :meth:`run`.

        """
        pass


class ModShift(BaseVetter):
    """Modshift vetter."""

    def __init__(self, lc_name="flux"):
        """
        Parameters
        ----------
        lc_name : str
            Name of the flux array in the ``lightkurve`` object.

         Attributes
        ----------
        time : array
            Time values of the TCE, populated by :meth:`run`.

        flux : array
            Flux values of the TCE, populated by :meth:`run`.

        period_days : float
            period of the TCE in days, populated by :meth:`run`.

        epoch_days : float
            epoch of the TCE in days, populated by :meth:`run`.

        duration_hrs : float
            transit duration of the TCE in hours, populated by :meth:`run`.

        box : astropy.units.Quantity object
            Flux from boxcar model of the TCE, populated by :meth:`run`.

        metrics : dict
            modshift result dictionary populated by :meth:`run`.
        """

        self.lc_name = lc_name
        self.time = None
        self.flux = None
        self.period_days = None
        self.epoch_days = None
        self.duration_hrs = None
        self.box = None
        self.metrics = None

    def run(self, tce, lightcurve, plot=False):
        """
        Runs modshift.compute_modeshift_metrics to populate the vetter object.

        Parameters
        -----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth

        lightcurve : lightkurve object
            lightkurve object with the time and flux to use for vetting.

        plot: bool
            option to show plot when initialy populating the metrics.
            Same as using the plot() method.

        Returns
        ------------
        metrics : dict
            modshift result dictionary containing the following:
                pri : primary signal
                sec : secondary signal
                ter : tertiary signal
                pos : largest positive event
                false_alarm_threshold : threshold for the 1 sigma false alarm
                Fred : red noise level, std(convolution) divided by std(lightcurve)
        """

        self.time, self.flux, time_offset_str = \
            lightkurve_utils.unpack_lk_version(lightcurve, self.lc_name)

        time_offset_q = const.string_to_offset[time_offset_str]

        self.flux = utils.set_median_flux_to_zero(self.flux)

        self.period_days = tce["period"].to_value(u.day)
        self.epoch_days = tce.get_epoch(time_offset_q).to_value(u.day)
        self.duration_hrs = tce["duration"].to_value(u.hour)

        self.box = model.create_box_model_for_tce(tce, self.time * u.day, time_offset_q)
        self.metrics, conv = modshift.compute_modshift_metrics(
            self.time,
            self.flux,
            self.box,
            self.period_days,
            self.epoch_days,
            self.duration_hrs,
            show_plot=plot,
        )

        return self.metrics

    def plot(self):
        met, c = modshift.compute_modshift_metrics(
            self.time,
            self.flux,
            self.box,
            self.period_days,
            self.epoch_days,
            self.duration_hrs,
            show_plot=True,
        )


class Lpp(BaseVetter):
    """LPP vetter."""

    def __init__(self, map_filename=None, lc_name="flux"):
        """
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

        metrics : dict
            lpp result dictionary populated by :meth:`run`.
        """
        self.lc_name = lc_name
        self.map_info = lpp.Loadmap(filename=map_filename)
        self.tce = None
        self.lc = None
        self.norm_lpp = None
        self.raw_lpp = None
        self.plot_data = None
        self.metrics = None

    def run(self, tce, lightcurve, plot=False):  
        """
        Runs lpp.compute_lpp_Transitmetric to populate the vetter object.

        Parameters
        -----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth

        lightcurve : lightkurve object
            lightkurve object with the time and flux to use for vetting.

        plot: bool
            option to show plot when initialy populating the metrics.
            Same as using the plot() method.

        Returns
        ------------
        metrics : dict
            lpp result dictionary containing the following:
                raw_lpp : Raw LPP value
                norm_lpp : LPP value normalized by period and SNR
                plot_data : The folded, binned transit prior to the LPP transformation
        """
        self.tce = tce
        self.lc = lightcurve

        self.lpp_data = lpp.Lppdata(self.tce, self.lc, self.lc_name)

        self.norm_lpp, self.raw_lpp, self.plot_data = lpp.compute_lpp_Transitmetric(  # noqa
            self.lpp_data, self.map_info
        )  # noqa: E501

        if plot:  # Added to allow plotting with run MD 2023
            target = self.tce.get("target_name", "Target")
            lpp.plot_lpp_diagnostic(self.plot_data, target, self.norm_lpp)

        self.metrics = {
            "raw_lpp": self.raw_lpp,
            "norm_lpp": self.norm_lpp,
            "plot_data": self.plot_data,
        }

        return self.metrics

    def plot(self):  # pragma: no cover
        if self.plot_data is None:
            raise ValueError("LPP plot data is empty. Execute self.run(...) first.")

        # target is populated in TCE, assume it already exists.
        target = self.tce.get("target_name", "Target")
        lpp.plot_lpp_diagnostic(self.plot_data, target, self.norm_lpp)


class OddEven(BaseVetter):
    """OddEven vetter"""

    def __init__(self, lc_name="flux", dur_frac=0.3):
        """
        Parameters
        ----------
        lc_name : str
            Name of the flux array in the ``lightkurve`` object.

        dur_frac:
            Fraction of in-transit duration to use for depth calculation.

        Attributes
        ------------
        odd_depth : tuple
            depth and error on depth of the odd transits, populated by :meth:`run`.

        even_depth : tuple
            depth and error on depth of the even transits, populated by :meth:`run`.

        oe_sigma : astropy.utils.masked.core.MaskedNDArray
            significance of difference of odd/even depth measurements, populated by :meth:`run`.

        metrics : dict
            modshift result dictionary populated by :meth:`run`.
        """

        self.lc_name = lc_name
        self.dur_frac = dur_frac
        self.odd_depth = None
        self.even_depth = None
        self.oe_sigma = None
        self.metrics = None

    def run(self, tce, lightcurve, plot=False):
        """
        Runs odd_even.calc_odd_even to populate the vetter object.

        Parameters
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth

        lightcurve : lightkurve object
            lightkurve object with the time and flux to use for vetting.

        plot: bool
            option to show plot when initialy populating the metrics.
            Same as using the plot() method.

        Returns
        ------------
        metrics : dict
            odd_even result dictionary containing the following:
                oe_sigma : significance of difference of odd/even depth measurements
                odd_depth : depth and error on depth of the odd transits
                even_depth : depth and error on depth of the even transits
        """

        self.time, self.flux, time_offset_str = lightkurve_utils.unpack_lk_version(  # noqa
            lightcurve, self.lc_name
        )

        time_offset_q = getattr(exo_const, time_offset_str)

        self.period = tce["period"].to_value(u.day)
        self.duration = tce["duration"].to_value(u.day)
        self.epoch = tce.get_epoch(time_offset_q).to_value(u.day)

        self.oe_sigma, self.odd_depth, self.even_depth = odd_even.calc_odd_even(  # noqa
            self.time,
            self.flux,
            self.period,
            self.epoch,
            self.duration,
            ingress=None,
            dur_frac=self.dur_frac,
        )

        self.metrics = {
            "oe_sigma": self.oe_sigma,
            "odd_depth": self.odd_depth,
            "even_depth": self.even_depth,
        }

        if plot:
            odd_even.diagnostic_plot(
                self.time,
                self.flux,
                self.period,
                self.epoch,
                self.duration * self.dur_frac,
                self.odd_depth,
                self.even_depth,
            )

        return self.metrics

    def plot(self):  # pragma: no cover
        odd_even.diagnostic_plot(
            self.time,
            self.flux,
            self.period,
            self.epoch,
            self.duration * self.dur_frac,
            self.odd_depth,
            self.even_depth,
        )


class TransitPhaseCoverage(BaseVetter):
    """Transit Phase Coverage Vetter"""

    def __init__(self, lc_name="flux", nbins=10, ndur=2):
        """
        Parameters
        ----------
        lc_name : str
            Name of the flux array in the ``lightkurve`` object.

        nbins : integer
            number bins to divide-up the in transit points. default is 10, giving
            an accuracy of 0.1.

        ndur : float
            the code considers a phase that cover ndur * transit_duration as
            "in transit". Default is 2

        Attributes
        ------------
        hist : array
            histogram of the times of length nbins, populated by :meth:`run`.

        bins : array
            corners of the bins for the histogram, length of nbins+1,
            populated by :meth:`run`.

        metrics : dict
            TransitPhaseCoverage result dictionary populated by :meth:`run`.
        """
        self.lc_name = lc_name
        self.nbins = nbins
        self.ndur = ndur
        self.hist = None
        self.bins = None
        self.metrics = None

    def run(self, tce, lightcurve, plot=False):
        """Runs transit_coverage.calc_coverage to populate the vetter object.

        Parameters
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        lightcurve : lightkurve object
            lightkurve object with the time and flux of the data to use for vetting.

        Returns
        ------------
        metrics : dict
            odd_even result dictionary containing the following:
                transit_phase_coverage : Fraction of coverage
        """
        time, flux, time_offset_str = lightkurve_utils.unpack_lk_version(
            lightcurve, self.lc_name
        )  # noqa: E50

        p_day = tce["period"].to_value(u.day)
        dur_hour = tce["duration"].to_value(u.hour)

        time_offset_q = getattr(exo_const, time_offset_str)
        epoch = tce.get_epoch(time_offset_q).to_value(u.day)

        tp_cover, self.hist, self.bins = transit_coverage.calc_coverage(
            time, p_day, epoch, dur_hour, ndur=self.ndur, nbins=self.nbins)

        self.metrics = {"transit_phase_coverage": tp_cover}

        if plot:
            transit_coverage.plot_coverage(self.hist, self.bins)

        return self.metrics

    def plot(self):  # pragma: no cover
        transit_coverage.plot_coverage(self.hist, self.bins)


class Sweet(BaseVetter):
    """SWEET Vetter"""

    def __init__(self, lc_name="flux", threshold_sigma=3):
        """
        Parameters
        ----------
        lc_name : str
            Name of the flux array in the ``lightkurve`` object.

        threshold_sigma : float
            Threshold for comparing signal to transit period.

        Attributes
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        lc : lightkurve object
            lightkurve object with the time and flux of the data to use for vetting.

        metrics : dict
            SWEET result dictionary populated by :meth:`run`.
        """

        self.lc_name = lc_name
        self.sweet_threshold_sigma = threshold_sigma
        self.tce = None
        self.lc = None
        self.metrics = None

    def run(self, tce, lightcurve, plot=False):
        """Runs sweet.sweet and sweet.construct_message to populate the vetter object.

        Parameters
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        lightcurve : lightkurve object
            lightkurve object with the time and flux of the data to use for vetting.

        plot: bool
            option to show plot when initialy populating the metrics.
            Same as using the plot() method.

        Returns
        ------------
        metrics : dict
            ``'msg'`` contains warnings, if applicable.
            ``'amp'`` contains the best fit amplitude, its uncertainty, and
            amplitude-to-uncertainty ratio for half-period, period, and
            twice the period.

        """
        self.tce = tce
        self.lc = lightcurve

        time, flux, time_offset_str = lightkurve_utils.unpack_lk_version(
            self.lc, self.lc_name
        )  # noqa: E50

        period_days = tce["period"].to_value(u.day)
        time_offset_q = getattr(exo_const, time_offset_str)
        epoch = tce.get_epoch(time_offset_q).to_value(u.day)
        duration_days = tce["duration"].to_value(u.day)

        result_dict = sweet.sweet(
            time, flux, period_days, epoch, duration_days, plot=plot
        )
        self.metrics = sweet.construct_message(result_dict, self.sweet_threshold_sigma)

        return self.metrics

    def plot(self):  # pragma: no cover
        self.run(self.tce, self.lc, plot=True)


class Centroid(BaseVetter):
    """Class to handle centroid vetting"""

    def __init__(self, lc_name="flux"):
        """
        Parameters
        ----------
        lc_name : str
            Name of the flux array in the ``lightkurve`` object.

        Attributes
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        tpf : obj
            ``lightkurve`` target pixel file object with pixels in column lc_name

        metrics : dict
            Centroid result dictionary populated by :meth:`run`."""

        self.lc_name = lc_name
        self.tce = None
        self.tpf = None
        self.metrics = None

    def run(self, tce, lk_tpf, plot=False):
        """Runs ent.compute_diff_image_centroids and cent.measure_centroid_shift
        to populate the vetter object.

        Parameters
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        lk_tpf: obj
            ``lightkurve`` target pixel file object with pixels in column lc_name

        plot: bool
            option to show plot when initialy populating the metrics.
            Same as using the plot() method.

        Returns
        ------------
        metrics : dict
            centroid result dictionary containing the following:
                offset : (float) Size of offset in pixels (or whatever unit centroids is in)
                significance : (float) The statistical significance of the transit.
                Values close to 1 mean the transit is likely on the target star.
                Values less than ~1e-3 suggest the target is not the source of the transit.
        """

        self.tce = tce
        self.tpf = lk_tpf

        time, cube, time_offset_str = lightkurve_utils.unpack_tpf(
            self.tpf, self.lc_name
        )  # noqa: E50

        period_days = tce["period"].to_value(u.day)
        time_offset_q = getattr(exo_const, time_offset_str)
        epoch = tce.get_epoch(time_offset_q).to_value(u.day)
        duration_days = tce["duration"].to_value(u.day)

        centroids, figs = cent.compute_diff_image_centroids(
            time, cube, period_days, epoch, duration_days, plot=plot
        )
        offset, signif, fig = cent.measure_centroid_shift(centroids, plot)
        figs.append(fig)

        # TODO: If plot=True, figs is a list of figure handles.
        # Do I save those figures, put them in a single pdf,
        # close them all?

        self.metrics = dict(offset=offset, significance=signif)
        return self.metrics

    def plot(self):  # pragma: no cover
        self.run(self.tce, self.tpf, plot=True)


class VizTransits(BaseVetter):
    """Class to return the number of transits that exist.
    It primarily plots all the transits on one figure along
    with a folded transit.
    """

    def __init__(self, lc_name="flux", max_transits=10, transit_only=False, 
                 smooth=10,):
        """
        Parameters
        ----------
        lc_name : str
            Name of the flux array in the ``lightkurve`` object.

        max_transits : type
            description

        transit_only : type
            description

        smooth : type
            description

         Attributes
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        metrics : dict
            VizTransits result dictionary populated by :meth:`run`.
        """

        self.lc_name = lc_name
        self.max_transits = max_transits
        self.transit_only = transit_only
        self.smooth = smooth
        self.tce = None
        self.metrics = None

    def run(self, tce, lightcurve, plot=False):
        """Runs viz_transits.plot_all_transits to populate the vetter object.

        Parameters
        ----------
        tce : tce object
            tce object is a dictionary that contains information about the tce
            to vet, like period, epoch, duration, depth.

        lightcurve : lightkurve object
            lightkurve object with the time and flux of the data to use for vetting.

        plot: bool
            option to show folded or unfolded plot.

        Returns
        ------------
        metrics : dict
            centroid result dictionary containing the following:
                num_transits : Number of transits with data in transit (3*duration).
        """

        time, flux, time_offset_str = lightkurve_utils.unpack_lk_version(
            lightcurve, self.lc_name)  # noqa: E50

        period_days = tce["period"].to_value(u.day)
        time_offset_q = getattr(exo_const, time_offset_str)
        epoch = tce.get_epoch(time_offset_q).to_value(u.day)
        duration_days = tce["duration"].to_value(u.day)
        depth = tce["depth"]

        n_has_data = viz_transits.plot_all_transits(time, flux, period_days,
                                                    epoch,
                                                    duration_days,
                                                    depth, max_transits=self.max_transits,
                                                    transit_only=self.transit_only,
                                                    plot=plot, units="d")

        viz_transits.plot_fold_transit(time, flux, period_days,
                                       epoch, depth, duration_days,
                                       smooth=self.smooth,
                                       transit_only=self.transit_only,
                                       plot=plot, units="d")

        self.metrics = {"num_transits": n_has_data}

        return self.metrics

    def plot(self, tce, lightcurve):

        _ = self.run(tce, lightcurve, max_transits=self.max_transits,
                     transit_only=self.transit_only, smooth=self.smooth,
                     plot=True)
