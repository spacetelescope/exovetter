from ipdb import set_trace as idebug

import exovetter.centroid.fastpsffit as fpf
import exovetter.centroid.disp as disp
import exovetter.utils as utils 

from pdb import set_trace as debug
import matplotlib.patches as mpatch 
import matplotlib.pyplot as plt
import numpy as np
    

def get_per_transit_diff_centroid(time, cube, period_days, epoch, duration_days, plot=False):

    #Turning off plotting for the moment
    plot=False 
    isnan = np.isnan(time)
    time = time[~isnan]
    cube = cube[~isnan]
    
    transits = getIngressEgressCadences(time, period_days, epoch, duration_days)
    
    #This might change to do per cadence diffs
    figs = []
    shifts = []
    for i in range(len(transits)):
        print("Transit %i" %(i))
        cin = transits[i]
        res, fig = measure_centroid_shift(cube, cin, plot=plot)
        shifts.append(res)
        figs.append(fig)

    return np.array(shifts)
    vetting_result, fig = centroid_vet(shifts, plot)
    figs.append(fig)
    
    if plot:
        return vetting_result, figs
    else:
        return vetting_result



def getIngressEgressCadences(time, period_days, epoch_btjd, duration_days):
    assert np.all(np.isfinite(time))

    idx = utils.mark_transit_cadences(time, period_days, epoch_btjd, duration_days)
    transits = np.array(utils.plateau(idx, .5))

    return transits


def centroid_vet(centroid_shifts, plot):
    centroid_shifts = np.array(centroid_shifts)
    
    #OOT - DIC
    dcol = centroid_shifts[:,-2] - centroid_shifts[:,0]
    drow = centroid_shifts[:,-1] - centroid_shifts[:,1]
    
    """
    This should use the code in covar. 
    Return offset, prob of this offset under null hypothesis.
    Produce a plot if necessary 
    
Futurework: Overlay the plot on a POSS plateau
    
    """
    #Placeholder algorithm 
    col0 = np.mean(dcol) 
    col_unc = np.std(dcol)

    row0 = np.mean(drow) 
    row_unc = np.std(drow)
    
    offset = np.hypot(col0, row0)
    doffset = np.hypot(col_unc, row_unc)

    if plot:
        fig = plot_centroid_vetting(col0, row0, col_unc, row_unc)

    #signif = offset / doffset
    else:
        fig = None 
        
    return [offset, doffset], fig


def plot_centroid_vetting(col0, row0, col_unc, row_unc):
    fig = plt.figure()
    
    plt.plot(col0, row0, 'ko', ms=12)
    for i in range(1,4):
        patch = mpatch.Ellipse([col0, row0], i*col_unc, i*row_unc, color='grey', alpha=.4)
        plt.gca().add_patch(patch)

        
    plt.axhline(0)
    plt.axvline(0)
    #plt.plot(dcol, drow, 'g^')
    plt.xlabel("Column Offset (pixels)")
    plt.ylabel("Row Offset (pixels)")
    plt.title("Mean Centroid Shift")
    return fig

def measure_centroid_shift(cube, cin, plot=False):
    oot, intrans, diff, ax = generateDiffImg(cube, cin, plot=plot)
    plt.pause(.01)

    guess = pickInitialGuess(oot)
    ootSoln = fpf.fastGaussianPrfFit(oot, guess)

    guess = pickInitialGuess(diff)
    diffSoln = fpf.fastGaussianPrfFit(diff, guess)

    guess = pickInitialGuess(intrans)
    intransSoln = fpf.fastGaussianPrfFit(intrans, guess)

    if not np.all( map(lambda x: x.success, [ootSoln, diffSoln, intransSoln]) ):
        print("WARN: Not all fits converged for [%i, %i]" %(cin[0], cin[1]))

    if plot:
        #Add fit results to axis
        #idebug()
        disp.plotCentroidLocation(diffSoln, marker='^')
        disp.plotCentroidLocation(ootSoln, marker='o')
        disp.plotCentroidLocation(intransSoln, marker='+')

        
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

    Also generates an image for each the $n$ cadedences before and after the transit,
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
        The difference between the flux in-transit and the average of the flux before and after

    Notes
    ---------
    When there is image motion, the before and after images won't be identical, and the difference
    image will show distinct departures from the ideal prf.
    """

    dur  = transits[1] - transits[0]
    s0, s1 = transits - dur
    e0, e1 = transits + dur

    before = cube[s0:s1].sum(axis=0)
    during = cube[transits[0]:transits[1]].sum(axis=0)
    after = cube[e0:e1].sum(axis=0)

    oot = .5 * (before + after)
    diff = oot - during

    if plot:
        fig = plt.figure()
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
    r0, c0 = np.unravel_index( np.argmax(img), img.shape)

    guess = [c0+.5, r0+.5, .5, 8*np.max(img), np.median(img)]
#    guess = [c0+.5, r0+.5, .5, 1*np.max(img), np.median(img)]
    return guess
