from exovetter.centroid import centroid as cent
import exovetter.centroid.fastpsffit as fpf
import exovetter.centroid.disp as disp
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip


def get_tpf_avgs(time, cube, period_days, epoch, duration_days, sector, sigma=None):
    """Given a TPF return the average OOT, INTRANS, and DIFF images"""

    isnan = np.isnan(time)
    time = time[~isnan]
    cube = cube[~isnan]

    transits = cent.getIngressEgressCadences(time, period_days, epoch, duration_days)

    oots = []
    intranits = []
    diffs = []

    for i in range(len(transits)):
        cin = transits[i]
        oot, intrans, diff = generateDiffImg(cube, cin)
        
        oots.append(oot)
        intranits.append(intrans)
        diffs.append(diff)

    # sometimes one or more of the transit images can be much brighter than the others, apply a sigma clipping and remove that transit from the stack
    if sigma:
        diff_medians = []
        for diff in diffs:
            diff_medians.append(np.median(diff))
        
        filtered_diffs = sigma_clip(diff_medians, sigma=sigma)
        
        oots = list(np.array(oots)[np.invert(filtered_diffs.mask)])
        intranits = list(np.array(intranits)[np.invert(filtered_diffs.mask)])
        diffs = list(np.array(diffs)[np.invert(filtered_diffs.mask)])

        for i, transit in enumerate(filtered_diffs.mask):
            if transit:
                print(f'removed transit {i+1} in sector {sector} from stack')

    avg_oot = np.sum(oots, axis=0)/len(diffs)
    avg_intrans = np.sum(intranits, axis=0)/len(diffs)
    avg_diff = np.sum(diffs, axis=0)/len(diffs)

    return avg_oot, avg_intrans, avg_diff

def compute_diff_image_centroids(sectors_oot, sectors_intrans, sectors_diff, sectors, max_oot_shift_pix=1.5, plot=False):
    """Compute difference image centroid shifts for every sector in a set of tpfs.

    Similar to centroid.centroid.compute_diff_image_centroids() but takes oot, intrans, and diff arrays as input
    
    Format of centroids array is:
    * Out of transit (OOT) centroid column
    * OOT row.
    * In Transit (ITR) column
    * ITR row
    * Difference image centroid (DIC) column
    * DIC row
    * DIC flag. A non-zero value means the centroid is untrustworthy."""

    centroids = []

    for i in range(len(sectors_oot)):
        if plot == True:
            fig = plt.figure()
            fig.set_size_inches(16, 4)
            disp.plotTransit(fig, sectors_oot[i], sectors_intrans[i], sectors_diff[i])
            plt.gcf().suptitle(f'Sector {sectors[i]} TPF')

        cents = measure_centroids(sectors_oot[i], sectors_intrans[i], sectors_diff[i], max_oot_shift_pix=max_oot_shift_pix, plot=plot)

        centroids.append(cents)
            
    centroids = np.array(centroids)
    all_sectors = list(np.arange(len(sectors_oot)))
    return centroids, all_sectors

def measure_centroids(avg_oot, avg_intrans, avg_diff, max_oot_shift_pix=0.5, starloc_pix = None, plot=False):
    """Using computed average oot, intrans, and diff images compute centroids as in centroid.centroid.measure_centroids()""" 

    # Constrain fit to within +-1 pixel for oot and intrans if desired.
    nr, nc = avg_oot.shape
    
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

    guess = cent.pickInitialGuess(avg_oot)
    ootSoln = fpf.fastGaussianPrfFit(avg_oot, guess, bounds=bounds)

    guess = cent.pickInitialGuess(avg_diff)
    diffSoln = fpf.fastGaussianPrfFit(avg_diff, guess)

    guess = cent.pickInitialGuess(avg_intrans)
    intransSoln = fpf.fastGaussianPrfFit(avg_intrans, guess, bounds=bounds)

    if not np.all(map(lambda x: x.success, [ootSoln, diffSoln, intransSoln])):
        print("WARN: Not all fits converged for [%i, %i]" % (cin[0], cin[1]))

    if plot:
        clr = "orange"
        if diffSoln.success:
            clr = "green"

        fig = plt.gcf()
        axlist = fig.axes 
        #assert len(axlist) == 3, axlist

        res = diffSoln.x
        for ax in axlist:
            if ax.get_label() == '<colorbar>':
                continue

            plt.sca(ax)
            disp.plotCentroidLocation(res[0], res[1], marker="^", color=clr,
                                    label="diff")

            res1 = ootSoln.x
            disp.plotCentroidLocation(res1[0], res1[1], marker="o", color=clr,
                                    label="OOT")

            res2 = intransSoln.x
            disp.plotCentroidLocation(res2[0], res2[1], marker="+", color=clr,
                                    label="InT")
            
            if starloc_pix is not None:
                disp.plotCentroidLocation(starloc_pix[0], starloc_pix[1], marker="*",
                                        color='red', label="Cat", ms=10)
            
        plt.legend(fontsize=12, framealpha=0.7, facecolor='silver')

    out = []
    out.extend(ootSoln.x[:2])
    out.extend(intransSoln.x[:2])
    out.extend(diffSoln.x[:2])
    flag = 0
    if not diffSoln.success:
        flag = 1
    if diffSoln.x[3] < 4 * np.median(avg_diff):
        flag = 2
    out.append(flag)

    return out

def generateDiffImg(cube, transits):
    """Generate a difference image for each transit"""

    dur = transits[1] - transits[0]
    s0, s1 = transits - dur
    e0, e1 = transits + dur

    before = cube[s0:s1].sum(axis=0)
    during = cube[transits[0]: transits[1]].sum(axis=0)
    after = cube[e0:e1].sum(axis=0)

    oot = 0.5 * (before + after)
    diff = oot - during

    return oot, during, diff