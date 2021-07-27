import exovetter.centroid.fastpsffit as fpf
import exovetter.centroid.covar as covar
import exovetter.centroid.disp as disp
import exovetter.utils as utils
import matplotlib.pyplot as plt
import numpy as np


def compute_diff_image_centroids(
        time, 
        cube, 
        period_days, 
        epoch, 
        duration_days, 
        max_oot_shift_pix=1.5,
        plot=False
):
    """Compute difference image centroid shifts for every transit in a dataset.

    Given a data cube containing a time-series of images, and a transit
    defined by a period, epoch and duration, compute centroid shift
    between in- and out-of- transit images for each transit covered by
    the time-series.


    Inputs
    -----------
    time
        (1d np array) Times of each slice of the data cube. Units of days

    cube
        (3d np array). Shape of the cube is (numCadences, numRows, numCols)
        There are numCadence images, and each image has a shape of
        (numRows, numCols).

    period_days
        (float) Orbital period of transit.
    epoch
        (float) Epoch of transit centre in the same time system as `time`.
    duration_days
        (float) Duration of transit.
    max_oot_shift_pix
        (float) Passed to `fastpsffit.fastGaussianPrfFit()

    Returns
    ---------------
    A 2d numpy array. Each row represents a single transit event.
    The columns are

    * Out of transit (OOT) centroid column
    * OOT row.
    * In Transit (ITR) column
    * ITR row
    * Difference image centroid (DIC) column
    * DIC row
    * DIC flag. A non-zero value means the centroid is untrustworthy.


    ITR images are computed by co-adding all cadences in transit
    (as defined by the period, epoch, and duration).
    OOT images are computed by co-adding 1 transit-duration worth
    of images from both before and after the transit.
    Difference image centroids (DIC) are computed by subtracting
    OOT from In-transit.
    """

    isnan = np.isnan(time)
    time = time[~isnan]
    cube = cube[~isnan]

    transits = getIngressEgressCadences(
        time, period_days, epoch, duration_days)

    figs = []
    centroids = []
    for i in range(len(transits)):
        cin = transits[i]
        cents, fig = measure_centroids(
            cube, 
            cin, 
            max_oot_shift_pix=max_oot_shift_pix,
            plot=plot
        )
        centroids.append(cents)
        figs.append(fig)
    centroids = np.array(centroids)
    return centroids, figs


def measure_centroid_shift(centroids, plot=False):
    """Measure the average offset of the DIC centroids from the OOT centroids.

    Inputs
    ----------
    centroids
        (2d np array) Output of :func:`compute_diff_image_centroids`

    Returns
    -----------
    offset
        (float) Size of offset in pixels (or whatever unit `centroids`
        is in)
    signif
        (float) The statistical significance of the transit. Values
        close to 1 mean the transit is likely on the target star.
        Values less than ~1e-3 suggest the target is not the
        source of the transit.
    fig
        A figure handle. Is **None** if plot is **False**
    """

    # DIC - OOT
    # dcol = centroids[:, 5] - centroids[:, 0]
    # drow = centroids[:, 4] - centroids[:, 1]
    dcol = centroids[:, 4] - centroids[:, 0]
    drow = centroids[:, 5] - centroids[:, 1]

    flags = centroids[:, -1].astype(bool)

    offset_pix, signif = covar.compute_offset_and_signif(
        dcol[~flags], drow[~flags])

    fig = None
    if plot:
        fig = covar.diagnostic_plot(dcol, drow, flags)
    return offset_pix, signif, fig


def getIngressEgressCadences(time, period_days, epoch_btjd, duration_days):
    assert np.all(np.isfinite(time))

    idx = utils.mark_transit_cadences(
        time, period_days, epoch_btjd, duration_days)
    transits = np.array(utils.plateau(idx, 0.5))

    return transits


def measure_centroids(cube, cin, max_oot_shift_pix=0.5, plot=False):
    """Private function of :func:`compute_diff_image_centroids`

    Computes OOT, ITR and diff images for a single transit event,
    and computes image centroid by fitting a Gaussian.

    Inputs
    ---------
    cube
        3d numpy array: Timeseries of images.
    cin
        2-tuple) Cadences of start and end of transit.
    max_oot_shift_pixel
        (float) OOT centroid is constrained in the fit to be within this distance
        of the centre of the postage stamp image
    plot
        True if a plot should be produced

    """

    oot, intrans, diff, ax = generateDiffImg(cube, cin, plot=plot)

    # Constrain fit to within +-1 pixel for oot and intrans if desired.
    nr, nc = oot.shape
    
    #Silently pin max shift to size of postage stamp
    max_oot_shift_pix = min(max_oot_shift_pix, nc/2, nr/2)
    
    #Short names for easier reading
    c2 = nc/2
    r2 = nr/2
    ms = max_oot_shift_pix
    
    bounds = [
        (c2-ms, c2+ms),
        (r2-ms, r2+ms),
        (0.2, 1),
        (None, None),
        (None, None),
    ]

    guess = pickInitialGuess(oot)
    ootSoln = fpf.fastGaussianPrfFit(oot, guess, bounds=bounds)

    guess = pickInitialGuess(diff)
    diffSoln = fpf.fastGaussianPrfFit(diff, guess)

    guess = pickInitialGuess(intrans)
    intransSoln = fpf.fastGaussianPrfFit(intrans, guess, bounds=bounds)

    if not np.all(map(lambda x: x.success, [ootSoln, diffSoln, intransSoln])):
        print("WARN: Not all fits converged for [%i, %i]" % (cin[0], cin[1]))

    if plot:
        clr = "orange"
        if diffSoln.success:
            clr = "green"

        res = diffSoln.x
        disp.plotCentroidLocation(res[0], res[1], marker="^", color=clr,
                                  label="diff")

        res = ootSoln.x
        disp.plotCentroidLocation(res[0], res[1], marker="o", color=clr,
                                  label="OOT")

        res = intransSoln.x
        disp.plotCentroidLocation(res[0], res[1], marker="+", color=clr,
                                  label="InT")
        plt.legend(fontsize=12, framealpha=0.7, facecolor='silver')

    out = []
    out.extend(ootSoln.x[:2])
    out.extend(intransSoln.x[:2])
    out.extend(diffSoln.x[:2])
    flag = 0
    if not diffSoln.success:
        flag = 1
    if diffSoln.x[3] < 4 * np.median(diff):
        flag = 2
    out.append(flag)

    return out, ax


def generateDiffImg(cube, transits, plot=False):
    """Generate a difference image.

    Also generates an image for each the $n$ cadedences before
    and after the transit,
    where $n$ is the number of cadences of the transit itself

    Inputs
    ------------
    cube
        (np 3 array) Datacube of postage stamps
    transits
        (2-tuples) Indices of the first and last cadence

    Optional Inputs
    -----------------
    plot
        (Bool) If true, generate a diagnostic plot


    Returns
    -------------
    Three 2d images, and a figure handle

    diff
        The difference between the flux in-transit and the average of the
        flux before and after

    Notes
    ---------
    When there is image motion, the before and after images won't be
    identical, and the difference
    image will show distinct departures from the ideal prf.
    """

    dur = transits[1] - transits[0]
    s0, s1 = transits - dur
    e0, e1 = transits + dur

    before = cube[s0:s1].sum(axis=0)
    during = cube[transits[0]: transits[1]].sum(axis=0)
    after = cube[e0:e1].sum(axis=0)

    oot = 0.5 * (before + after)
    diff = oot - during

    if plot:
        fig = plt.figure()
        fig.set_size_inches(16, 4)
        disp.plotTransit(fig, oot, during, diff)
    else:
        fig = None

    return oot, during, diff, fig


def pickInitialGuess(img):
    """Pick initial guess of params for `fastGaussianPrfFit`

    Inputs
    ---------
    img
        (2d np array) Image to be fit

    Returns
    ---------
    An array of initial conditions for the fit
    """
    r0, c0 = np.unravel_index(np.argmax(img), img.shape)

    guess = [c0 + 0.5, r0 + 0.5, 0.5, 8 * np.max(img), np.median(img)]
    return guess
