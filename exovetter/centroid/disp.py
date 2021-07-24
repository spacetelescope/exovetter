from __future__ import print_function
from __future__ import division

import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np


def plotTransit(fig, oot, during, diff, **kwargs):

    fig.add_subplot(131)
    plotImage(oot, **kwargs)
    plt.title("OOT")

    fig.add_subplot(132)
    plotImage(during, **kwargs)
    plt.title("In-transit")

    fig.add_subplot(133)
    plotDiffImage(diff, **kwargs)
    plt.title("Difference")


def plotImage(img, **kwargs):
    """Plot an image in linear scale

    Inputs
    --------
    img
        (2d np array) Image to plot

    Optional Inputs
    -----------------
    log
        (bool) Plot the image in log scalings. (Default False)
    origin
        (str) If 'bottom', put origin of image in bottom left
        hand corner (default).
        If 'top', but it in top left corner

    interpolation
        (str) Interpolation method. Default is nearest. See `plt.imshow`
        for more options

    cmap
        (plt.cm.cmap) Color map. Default is YlGnBu_r

    extent
        (4-tuple) Extent of image. See `plt.imshow` for more details

    All other optional arguments passed to `plt.imshow`


    Returns
    ----------
    **None**

    Output
    -------
    A plot is returned
    """
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    if "cmap" not in kwargs:
        kwargs["cmap"] = plt.cm.YlGnBu_r

    if "norm" not in kwargs:
        kwargs["norm"] = mcolor.Normalize()

    if "extent" not in kwargs:
        shape = img.shape
        extent = [0, shape[1], 0, shape[0]]
        kwargs["extent"] = extent

    showValues = kwargs.pop("showValues", False)
    log = kwargs.pop("log", False)

    if log:
        img = img.copy()
        mn = np.min(img)
        if mn < 0:
            offset = -1.1 * mn
            img += offset
        img = np.log10(img)

    plt.imshow(img, **kwargs)

    if showValues:
        showPixelValues(img, kwargs["cmap"], kwargs["norm"])
    plt.colorbar()


def showPixelValues(img, cmap, norm):

    fmt = "%i"
    nr, nc = img.shape
    for i in range(nc):
        for j in range(nr):
            clr = cmap(norm(img[j, i]))

            textcolor = "w"
            if np.prod(clr) > 0.2:
                textcolor = "k"

            txt = fmt % (img[j, i])
            plt.text(i + 0.5, j + 0.5, txt, color=textcolor, ha="center")


def plotDifferenceImage(img, **kwargs):
    """Plot a difference image.

    The colour bar is chosen so zero flux is at the centre of the colour map
    """
    if "origin" not in kwargs:
        kwargs["origin"] = "lower"

    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "nearest"

    if "cmap" not in kwargs:
        kwargs["cmap"] = plt.cm.RdBu_r

    if "extent" not in kwargs:
        shape = img.shape
        extent = [0, shape[1], 0, shape[0]]
        kwargs["extent"] = extent

    plt.imshow(img, **kwargs)
    plt.colorbar()
    vm = max(np.fabs([np.min(img), np.max(img)]))
    plt.clim(-vm, vm)


def plotDiffImage(img, **kwargs):
    """Mneumonic"""
    return plotDifferenceImage(img, **kwargs)


def plotCentroidLocation(col, row, **kwargs):
    """Add a point to the a plot.

    Private function of `generateDiffImgPlot()`
    """
    ms = kwargs.pop("ms", 9)
    clr = kwargs.pop("color", "g")
    mec = kwargs.pop("mec", "white")
    mew = kwargs.pop("mew", "1")
    marker = kwargs.pop("marker", "o")

    #plt.plot([col], [row], ms=ms + 1, **kwargs)

    plt.plot([col], [row], marker=marker, color=clr, mec=mec,
             mew=mew, ms=ms, lw=0, **kwargs)
