"""Module to handle Locality Preserving Projections (LPP)."""

# TODO: Replace camelcase and populate API docstrings.
# TODO: Maybe the boat has sailed but maybe some of these functions,
# especially the ones that pass around mapInfo should have been class methods.

from exovetter import lightkurve_utils
import copy
import warnings

import numpy as np
from astropy import units as u
from astropy.utils.data import download_file, _is_url
from scipy import io as spio

__all__ = ['compute_lpp_Transitmetric', 'runningMedian', 'foldBinLightCurve',
           'computeRawLPPTransitMetric', 'knnDistance_fromKnown',
           'periodNormalLPPTransitMetric', 'lpp_onetransit',
           'lpp_averageIndivTransit', 'plot_lpp_diagnostic', 'Lppdata',
           'Loadmap']


def compute_lpp_Transitmetric(data, mapInfo):
    """Take a data class with light curve info
    and the mapInfo with information about the mapping to use.
    It then returns a lpp metric value.

    """
    binFlux, binPhase = foldBinLightCurve(data, mapInfo.ntrfr, mapInfo.npts)

    plot_data = dict()
    plot_data['bin_flux'] = binFlux
    plot_data['bin_phase'] = binPhase

    # Dimensionality Reduction and knn parts
    rawTLpp, transformedTransit = computeRawLPPTransitMetric(binFlux, mapInfo)

    # Normalize by Period Dependence
    normTLpp = periodNormalLPPTransitMetric(
        rawTLpp, np.array([data.period, data.mes]), mapInfo)

    plot_data['lpp_transform'] = transformedTransit

    return normTLpp, rawTLpp, plot_data


def runningMedian(t, y, dt, runt):
    """Take a running median of size dt. Return values at times given in runt.

    """
    newy = np.zeros(len(y))
    newt = np.zeros(len(y))

    srt = np.argsort(t)
    newt = t[srt]
    newy = y[srt]

    runy = []
    for i in range(len(runt)):
        tmp = []
        for j in range(len(newt)):
            if (newt[j] >= (runt[i] - dt)) and (newt[j] <= (runt[i] + dt)):
                tmp.append(newy[j])

        if np.isnan(np.nanmedian(np.array(tmp))):
            runy.append(0)
        else:
            runy.append(np.nanmedian(np.array(tmp)))

    return list(runt), runy


def foldBinLightCurve(data, ntrfr, npts):
    """Fold and bin lightcurve for input to LPP metric calculation.

    Parameters
    ----------
    data
        Contains time, tzero, dur, period, mes and flux (centered around zero).

    ntrfr
        Number of transit fraction for binning around transit ~1.5.

    npts
        Number of points in the final binning.

    """
    # Create a phased light curve
    phaselc = np.mod((data.time - (data.tzero - 0.5 * data.period)) /
                     data.period, 1)
    flux = data.flux
    mes = data.mes

    # Determine the fraction of the time the planet transits the star.
    # Insist that ntrfr * transit fraction
    if ~np.isnan(data.dur) & (data.dur > 0):
        transit_dur = data.dur
    else:
        transit_dur = 0.2 * data.period / 24.

    transit_fr = transit_dur / 24. / data.period
    if (transit_fr * ntrfr) > 0.5:
        transit_fr = 0.5 / ntrfr

    # Specify the out of transit (a) and the in transit regions
    binover = 1.3
    if mes <= 20:
        binover = -(1 / 8.0) * mes + 3.8

    endfr = .03
    midfr = .11
    a = np.concatenate((np.arange(endfr, .5 - midfr, 1 / npts),
                        np.arange((0.5 + midfr), (1 - endfr), 1 / npts)),
                       axis=None)
    ovsamp = 4.0
    # bstep=(ovsamp*ntrfr*transit_fr)/npts
    b_num = 41
    b = np.linspace((0.5 - ntrfr * transit_fr),
                    (0.5 + ntrfr * transit_fr), b_num)

    [runta, runya] = runningMedian(phaselc, flux, binover / npts, a)
    [runtb, runyb] = runningMedian(
        phaselc, flux, (binover * ovsamp * ntrfr * transit_fr) / npts, b)

    # Combine the two sets of bins
    runymess = np.array(runya + runyb)
    runtmess = np.array(runta + runtb)

    srt = np.argsort(runtmess)
    runy = runymess[srt]
    runt = runtmess[srt]

    # Scale the flux by the depth so everything has the same depth.
    # Catch or dividing by zero is to not scale.
    scale = -1 * np.min(runyb)
    if scale != 0:
        scaledFlux = runy / scale
    else:
        scaledFlux = runy

    binnedFlux = scaledFlux
    phasebins = runt

    return binnedFlux, phasebins


def computeRawLPPTransitMetric(binFlux, mapInfo):
    """Perform the matrix transformation with LPP.
    Do the KNN test to get a raw LPP transit metric number.

    .. note:: This requires ``lpproj`` package.

    """
    # https://github.com/jakevdp/lpproj
    from lpproj import LocalityPreservingProjection

    Yorig = mapInfo.YmapMapped
    lpp = LocalityPreservingProjection(n_components=mapInfo.n_dim)
    lpp.projection_ = mapInfo.YmapM

    # To equate to Matlab LPP methods, we need to remove mean of transform.
    # Check if this is correct, YmapMean is an array that is transit shapped
    normBinFlux = binFlux - mapInfo.YmapMean

    inputY = lpp.transform(normBinFlux.reshape(1, -1))

    knownTransitsY = Yorig[mapInfo.knnGood, :]

    dist, ind = knnDistance_fromKnown(knownTransitsY, inputY, mapInfo.knn)

    rawLppTrMetric = np.mean(dist)

    return rawLppTrMetric, binFlux


def knnDistance_fromKnown(knownTransits, new, knn):
    """For a group of known transits and a new one,
    use KNN to determine how close the new one is to the known transits.

    .. note:: Using KNN ``minkowski p = 2 ()``. Requires ``scikit-learn``.

    """
    from sklearn.neighbors import NearestNeighbors

    # p=3 sets a minkowski distance of 3.
    # TODO: Check that you really used 3 for matlab.
    nbrs = NearestNeighbors(n_neighbors=int(knn), algorithm='kd_tree', p=2)
    nbrs.fit(knownTransits)

    distances, indices = nbrs.kneighbors(new)

    return distances, indices


def periodNormalLPPTransitMetric(rawTLpp, newPerMes, mapInfo):
    """Normalize the rawTransitMetric value by those with the closest period.
    This part removes the period dependence of the metric at short periods.
    Plus, it makes a value near one be the threshold between good and bad.

    ``newPerMes`` is the ``np.array([period, mes])`` of the new sample.

    """
    knownTrPeriods = mapInfo.mappedPeriods[mapInfo.knnGood]
    knownTrMes = mapInfo.mappedMes[mapInfo.knnGood]
    knownTrrawLpp = mapInfo.dymeans[mapInfo.knnGood]
    nPercentil = mapInfo.nPercentil
    nPsample = mapInfo.nPsample

    # Find the those with the nearest periods  Npsample-nneighbors
    logPeriods = np.log10(knownTrPeriods)
    logMes = np.log10(knownTrMes)
    knownPerMes = np.stack((logPeriods, logMes), axis=-1)

    np.shape(knownPerMes)
    logNew = np.log10(newPerMes).reshape(1, -1)
    # logNew=np.array([np.log10(newPeriod)]).reshape(1,1)

    dist, ind = knnDistance_fromKnown(knownPerMes, logNew, nPsample)

    # Find the nthPercentile of the rawLpp of these indicies
    nearPeriodLpp = knownTrrawLpp[ind]

    LppNPercentile = np.percentile(nearPeriodLpp, nPercentil)

    NormLppTransitMetric = rawTLpp / LppNPercentile

    return NormLppTransitMetric


def lpp_onetransit(tcedata, mapInfo, ntransit):
    """Chop down the full time series to one orbital period.
    Then gather the lpp value for that one transit.

    """
    startTime = tcedata.time[0] + ntransit * tcedata.period

    # A few cadences of overlap
    endTime = tcedata.time[0] + (ntransit + 1) * tcedata.period + 3 / 24.0

    want = (tcedata.time >= startTime) & (tcedata.time <= endTime)
    newtime = tcedata.time[want]
    newflux = tcedata.flux[want]

    nExpCad = (tcedata.time[-1] - tcedata.time[0]) / tcedata.period

    if len(newtime > nExpCad * 0.75):
        onetransit = copy.deepcopy(tcedata)
        onetransit.time = newtime
        onetransit.flux = newflux
        normTLpp, rawTLpp, transformedTr = compute_lpp_Transitmetric(
            onetransit, mapInfo)
    else:
        normTLpp = np.nan
        rawTLpp = np.nan

    return normTLpp, rawTLpp


def lpp_averageIndivTransit(tcedata, mapInfo):
    """Create the loop over individual transits and return
    array normalized lpp values, mean, and std.
    Input TCE object and mapInfo object.

    It is unclear that this individual transit approach
    separates out several new false positives.

    It probably would require re-tuning for low SNR signals.

    """
    length = tcedata.time[-1] - tcedata.time[0]
    ntransits = int(np.floor(length / tcedata.period))

    lppNorms = np.ones(ntransits)
    lppRaws = np.ones(ntransits)

    # nExpCad=(tcedata.time[-1]-tcedata.time[0])/tcedata.period

    for i in range(ntransits):
        lppNorms[i], lppRaws[i] = lpp_onetransit(tcedata, mapInfo, i)

    lppMed = np.nanmedian(lppNorms)
    lppStd = np.nanstd(lppNorms)

    return lppNorms, lppMed, lppStd, ntransits


def plot_lpp_diagnostic(data, target, norm_lpp):
    """Plot LPP data for diagnostics.

    .. note:: Requires ``matplotlib``.

    Parameters
    ----------
    data : dict
        Contains ``bin_flux`` and ``bin_phase`` for plotting.

    target : str
        Contains target name on the plot.

    norm_lpp : float
        Normalized LPP transit metric value.
        Used as string on the top of the plot.

    Returns
    -------
    fig : obj
        Matplotlib figure.

    """
    import matplotlib.pyplot as plt

    phase = data['bin_phase']
    flux = data['bin_flux']

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(phase, flux, 'b.', ms=5, label="LPP Bins")
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title(f"LPP Binning for {target}")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(212)
    ax2.plot(flux, 'k.', ms=5, label=f"LPP Norm = {norm_lpp:5.3f}")
    ax2.set_xlabel('Bin Number')
    ax2.legend()

    plt.draw()

    return fig


class Lppdata:
    """Class to handle LPP data.

    Parameters
    ----------
    tce : `~exovetter.tce.TCE`
        TCE object.

    lc : obj
        ``lightkurve`` object.

    """

    def __init__(self, tce, lc, lc_name="flux", default_snr=10.):
        # TODO: Needs a check lightcurve function

        self.check_tce(tce, default_snr)

        # FIXME: This looks more correct but fails the test.
        # from exovetter import const as exo_const
        # self.tzero = tce.get_epoch(
        #     getattr(exo_const, lc.time_format)).to_value(u.day)

        self.tzero = tce['epoch'].to_value(u.day)
        self.dur = tce['duration'].to_value(u.hr)
        self.period = tce['period'].to_value(u.day)

        self.mes = default_snr
        if 'snr' in tce.keys():
            self.mes = tce['snr']

        self.time, self.flux, _ = \
            lightkurve_utils.unpack_lk_version(lc, lc_name)

        # make sure flux is zero norm.
        if np.round(np.median(self.flux)) != 0:
            self.flux = self.flux - np.median(self.flux)

    def check_tce(self, tce, default_snr):
        """Validate TCE."""

        if 'period' not in tce.keys():
            raise KeyError('Period required for the TCE to run LPP.')

        if 'snr' not in tce.keys():
            warnings.warn('LPP requires a MES or SNR value stored as snr '
                          f'in the tce. Using a value of {default_snr}.')


class Loadmap:
    """Class to handle map info parsing.
    Read in MATLAB blob. Use the DV trained one by default.

    .. note:: Requires ``scipy``.

    Parameters
    ----------
    filename : str or `None`
        Full path to a LPP ``.mat`` file.
        If not provided, a built-in default is used.
        If URL is provided, it will be cached using :ref:`astropy:utils-data`

    """
    builtin_mat = 'https://stsci.box.com/shared/static/1ffi1t1fhae82d7xeqexw4ymlhlk0ov4.mat'  # noqa

    def __init__(self, filename=None):
        if filename is None:
            filename = self.builtin_mat

        if _is_url(filename):
            self.filename = download_file(filename, cache=True)
        else:
            self.filename = filename

        mat = spio.loadmat(self.filename, matlab_compatible=True)

        # Pull out the information we need.
        # FIXME: No mapInfoDV nor commented stuff below in SourceForge version
        key = 'mapInfoDV'
        # key = 'map'
        self.n_dim = mat[key]['nDim'][0][0][0][0]
        self.Ymap = mat[key]['Ymap'][0][0][0][0]
        self.YmapMapping = self.Ymap['mapping']
        self.YmapMean = self.YmapMapping['mean'][0][0][0]
        self.YmapM = self.YmapMapping['M'][0][0]
        self.YmapMapped = self.Ymap['mapped']
        self.knn = mat[key]['knn'][0][0][0][0]
        self.knnGood = mat[key]['knnGood'][0][0][:, 0]
        self.mappedPeriods = mat[key]['periods'][0][0][0]
        self.mappedMes = mat[key]['mes'][0][0][0]
        self.nPsample = mat[key]['nPsample'][0][0][0][0]  # number to sample  # noqa: E501
        self.nPercentil = mat[key]['npercentilTM'][0][0][0][0]
        self.dymeans = mat[key]['dymean'][0][0][0]
        self.ntrfr = 2.0
        self.npts = 80.0
