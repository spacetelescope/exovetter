from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import scipy.stats as spstats
import numpy as np

"""
    Given a set of points, (x,y), compute to properties of
    the probability density ellipse from which they were drawn.

    Given a set of points (x,y), assume they were drawn from
    a probability density distribution well described by a
    2-dimensional Gaussian. The distribution may show co-variance,
    (i.e large values of x may make large values of y more or less
    likely, and vice-versa), so in general, the distribution will
    not be aligned with the x and y-axes.

    Additionally, some of the input points are outliers, so must
    perform some kind of sigma clipping to remove those values from
    the fit.

    The probability density ellipse is descibed by

    * A vector pointing from the origin to the centre of the ellipse.
    * A vector from the centre pointing along the semi-major axis, with
      length equal to the the :math:`\\sigma` of the Gaussian in
      that direction, :math:`\\sigma_a`.
    * A vector from the centre pointing along the semi-minor axis,
      with length equal to the :math:`\\sigma` of the Gaussian in
      that direction, :math:`\\sigma_b`.

    The algorith proceeds as follows.
    1. Input all data points (x,y)
    2. Compute the centroid, semi-major and semi-minor axes of the ellipse.
    3. Transform all the input points to a coordinate system where
       their first coordainte is distance along the semi-major axis in
       units of :math:`\\sigma_a`, and the second coordinate is the distance
       along the semi-minor axis in units of :math:`\\sigma_b`. This
       distrubtion is radiall symmetric around the origin.
    4. Compute the survival probability of each point, i.e the probability
       of finding a point at least that far from the origin in this
       transformed space.
    5. Reject points whose survival probability is less that some
       threshold. These points are the outliers
    6. Go back to step 2 and repeat until no more outliers found.
"""


def diagnostic_plot(x, y, flag=None):

    if flag is None:
        flag = np.zeros(len(x))
    idx = flag.astype(bool)

    # Turned off outlier detection because it doesn't work well.
    # idx = find_outliers(x, y, initial_clip=idx, threshold=1e-5)
    mu_x = np.mean(x[~idx])
    mu_y = np.mean(y[~idx])
    sma, smi = compute_eigen_vectors(x[~idx], y[~idx])

    plt.clf()
    plt.gcf().set_size_inches((10, 8))
    plt.plot(x, y, "ko", mec="w", label="Centroids", zorder=+5)
    if np.any(idx):
        plt.plot(
            x[idx],
            y[idx],
            "o",
            color="pink",
            label="Outliers",
            zorder=+6)

    # prob = compute_prob_of_points(x, y, sma, smi)
    for i in range(len(x)):
        plt.text(x[i], y[i], " %i" % (i), zorder=+5)

    sigma_a = np.linalg.norm(sma)
    sigma_b = np.linalg.norm(smi)
    angle_deg = np.degrees(np.arctan2(sma[1], sma[0]))

    ax = plt.gca()
    if 1:
        for p in [0.68, 0.95, 0.997]:
            scale = spstats.rayleigh().isf(1 - p)
            width = 2 * sigma_a * scale
            height = 2 * sigma_b * scale
            ell = Ellipse(
                [mu_x, mu_y],
                width=width,
                height=height,
                angle=angle_deg,
                color="gray",
                alpha=0.2,
                label="%g%% Prob" % (100 * p),
            )
            ax.add_patch(ell)

    plt.axhline(0)
    plt.axvline(0)
    plt.plot(0, 0, "*", color="c", ms=28, mec="w", mew=2)
    plt.xlabel("Column shift (pixels)")
    plt.ylabel("Row shift (pixels)")
    plt.axis("equal")
    plt.legend()

    offset, signif = compute_offset_and_signif(x[~idx], y[~idx])
    msg = "Offset %i pixels\nProb Transit on Target: %.0e" % (offset, signif)
    plt.title(msg)
    return plt.gcf()


def compute_offset_and_signif(col, row):
    """Compute the mean offset of a set of points from the origin

    Computes the mean position of the inputs, and the statistical significance
    of the offset. The statistical signifance calculation assumes the
    points are drawn from a 2d Gaussian. See module docs above.

    Inputs
    ----------
    col, row
        (1d np arrays) Column and row values for each obseration.

    Returns
    ----------
    A tuple of (offset, signif)
    The offset is the offset of the mean value for column and row from
    the origin, and is measured in the same units as the inputs.
    The significance
    is measured as the probability of seeing an offset at least this large
    in this direction, given the variance (and co-variances) of the
    column and row values. Values close to 1 indicate the offset is consistent
    with zero. Low values (< 1e-3 or so) indicate the offset is
    statistically significant
    """

    centroid = get_centroid_point(col, row)
    sma, smi = compute_eigen_vectors(col, row)

    offset_pixels = np.linalg.norm(centroid)
    prob = compute_prob_of_points([0], [0], sma, smi, centroid)
    return offset_pixels, prob


def find_outliers(x, y, threshold=1e-6, initial_clip=None, max_iter=10):
    """Find data points that are not outliers

    Use a sigma-clipping algorithm to identify outlier points
    in the distribution of points (x,y), then return the indices
    of the inliers, i.e the good data points.

    This can crash if everything gets clipped!
    """

    # idx is true if a point is bad
    idx = initial_clip
    if idx is None:
        idx = np.zeros(len(y), dtype=bool)

    assert len(x) == len(y)
    assert len(idx) == len(y)

    old_num_clipped = np.sum(idx)
    for i in range(int(max_iter)):
        sma, smi = compute_eigen_vectors(x[~idx], y[~idx])
        prob = compute_prob_of_points(x, y, sma, smi)
        print(prob)
        new_idx = idx | (prob < threshold)
        new_num_clipped = np.sum(new_idx)

        # print( "Iter %i: %i (%i) clipped points "
        # %(i, new_num_clipped, old_num_clipped))

        if new_num_clipped == old_num_clipped:
            return new_idx

        old_num_clipped = new_num_clipped
        idx = new_idx
        i += 1

    # Return indices of *good* points
    return new_idx


def compute_eigen_vectors(x, y):
    """Compute semi-major and semi-minor axes of the probability
    density ellipse.

    Proceeds by computing the eigen-vectors of the covariance matrix
    of x and y.

    Inputs
    -----------
    x, y
        (1d numpy arrays). x and y coordinates of points to fit.


    Returns
    ---------
    sma_vec
        (2 elt numpy array) Vector describing the semi-major axis
    smi_vec
        (2 elt numpy array) Vector describing the semi-minor axis
    """

    assert len(x) == len(y)
    cov = np.cov(x, y)
    assert np.all(np.isfinite(cov))
    eigenVals, eigenVecs = np.linalg.eig(cov)

    # idebug()
    sma_vec = eigenVecs[:, 0] * np.sqrt(eigenVals[0])
    smi_vec = eigenVecs[:, 1] * np.sqrt(eigenVals[1])
    return sma_vec, smi_vec


def compute_prob_of_points(x, y, sma_vec, smi_vec, cent_vec=None):
    """Compute the probability of observing points as far away as (x,y) for
    a given ellipse.

    For the ellipse described by centroid, semi-major and semi-minor axes
    `sma_vec` and `smi_vec`, compute the
    probability of observing points at least as far away as x,y in
    the direction of that point.

    If no cent_vec supplied, it is computed as the centroid of the
    input points.

    Inputs
    ---------
    x, y
        (1d numpy arrays). x and y coordinates of points to fit.
    sma_vec
        (2 elt numpy array) Vector describing the semi-major axis
    smi_vec
        (2 elt numpy array) Vector describing the semi-minor axis
    cent_vec
        (2 elt numpy array) Vector describing centroid of ellipse.
        If **None**, is set to the centroid of the input points.

    Returns
    --------
    1d numpy array of the probabilities for each point.
    """
    if cent_vec is None:
        cent_vec = get_centroid_point(x, y)

    assert len(x) == len(y)
    assert len(cent_vec) == 2
    assert len(sma_vec) == 2
    assert len(smi_vec) == 2

    xy = np.vstack([x, y]).transpose()
    rel_vec = xy - cent_vec

    # The covector of a vector **v** is defined here as a vector that
    # is parallel to **v**, but has length :math:`= 1/|v|`
    # Multiplying a vector by the covector of the semi-major axis
    # gives the projected distance of that vector along that axis.
    sma_covec = sma_vec / np.linalg.norm(sma_vec) ** 2
    smi_covec = smi_vec / np.linalg.norm(smi_vec) ** 2
    coeff1 = np.dot(rel_vec, sma_covec)  # Num sigma along major axis
    coeff2 = np.dot(rel_vec, smi_covec)  # Num sigma along minor axis

    dist_sigma = np.hypot(coeff1, coeff2)
    prob = spstats.rayleigh().sf(dist_sigma)
    return prob


def get_centroid_point(x, y):
    return np.array([np.mean(x), np.mean(y)])
