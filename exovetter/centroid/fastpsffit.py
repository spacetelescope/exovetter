# -*- coding: utf-8 -*-

"""
Created on Mon Nov 19 16:39:13 2018

A much faster PRF fitter, with the caveat that the psf model is hardcoded.

psffit.py can fit an arbitrary PSF model to an image.
The cost of this flexibility
is that it must perform numerical intergration to calculate
the flux in each pixel.
This is slow. On my test machine, a 10x12 image takes 20ms to compute.

Since by far the most common model to fit is that of a symmetric
Gaussian function with
a constant sky background, and this model can be computed quite
quickly, this module
enables this special case to be run much faster. On the same machine,
the same image
can be computed in 95.7us, or a x200 speed up. There's still more speed up to
be had if you make a Model() class that assigns memory for the model
once and overwrites
it each time instead of computing from scratch in each call.

The downside is that none of the code is shared with the general purpose code.
Efforts to use numba don't seem to help much for some reason


The only two public methods are
* fastGaussianPrfFit
* computeModel


@author: fergal
"""

import scipy.optimize as spOpt
from scipy.special import erf
import numpy as np


def fastGaussianPrfFit(img, guess, bounds=None):
    """Fit a Symmetric Gaussian PSF to an image, really quickly

    Inputs
    --------
    img
        (2d numpy array) Image to fit
    prfFunc
        (function) Model to fit. See module level documentation for more
        details.
    guess
        (tuple or array) Elements are

        col0, row0
            Location of PSF centroid
        sigma
            Width of gaussian
        flux
            Height of gaussian. Beware this is not normalized
        sky
            Background level
    bounds
        array of tuples indicating the bounds of the fit for
        col, row, sigma, flux, sky.
        Default allows fit to center anywhere on the image.

    Returns
    ------------
    A scipy.optiminze.ResultsObject. The .x attribute contains the best
    fit parameters
    """

    assert len(guess) == 5

    nr, nc = img.shape
    if bounds is None:
        bounds = [
            (0, nc),
            (0, nr),
            (0.2, 1),
            (None, None),
            (None, None),
        ]
        
    mask = None
    soln = spOpt.minimize(
        costFunc, guess, args=(img, mask), method="L-BFGS-B", bounds=bounds
    )
    return soln


def costFunc(arglist, img, mask=None):
    """Compute difference between image and its model for given model params

    Inputs
    ----------
    arglist
        (tuple or array) Tunable parameters of model
    func
        (function) Model to fit
    img
        (2d np array) Image to fit


    Optional Inputs
    ----------------
    mask
        (2d np array) Zero elements of mask indicate bad data which should
        not be
        included in the fit


    Returns
    ----------
    float
    """

    nr, nc = img.shape
    model = computeModel(nc, nr, arglist)
    diff = img - model

    if mask is not None:
        assert np.all(mask.shape == img.shape)
        diff[~mask] = 0
        img[~mask] = 0  # In case bad values are set to Nan

    cost = np.sqrt(np.sum(diff ** 2))
    return cost


def computeModel(numCols, numRows, arglist):
    """Compute model flux for an image with size (numCols, numRows)

    Inputs
    -------
    numCols, numRows
        (ints) Shape of the image to compute the model PRF for
    func
        (function) Model PRF
    arglist
        (tuple or array) Tunable parameters of the model

    Returns
    ----------
    A 2d numpy array representing the model PRF image.
    """

    model = np.zeros((numRows, numCols))

    xc = np.arange(numCols)
    xr = np.arange(numRows)
    cols, rows = np.meshgrid(xc, xr)

    model = analytic_gaussian_integral(cols, rows, *arglist)

    return model


def analytic_gaussian_integral(col, row, col0, row0, sigma0, flux0, sky):

    z_col1 = 0.5 * (col - col0) / sigma0
    z_col2 = 0.5 * (col + 1 - col0) / sigma0

    z_row1 = 0.5 * (row - row0) / sigma0
    z_row2 = 0.5 * (row + 1 - row0) / sigma0

    flux = flux0
    flux *= phi(z_col2) - phi(z_col1)
    flux *= phi(z_row2) - phi(z_row1)
    flux += sky
    return flux


# Precompute for speed
sqrt2 = np.sqrt(2)


def phi(z):
    """Compute integral of gaussian function in the range (-Inf, z],
    `z` is defined as (x - x0) / sigma, where x0 is the central value
    of the Gaussian.

    See `scipy.special.erf` for details
    """

    return 0.5 * (1 + erf(z / sqrt2))
