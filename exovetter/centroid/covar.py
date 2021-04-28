from ipdb import set_trace as idebug
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
      length equal to the the :math:`\sigma` of the Gaussian in 
      that direction, :math:`\sigma_a`. 
    * A vector from the centre pointing along the semi-minor axis, 
      with length equal to the :math:`\sigma` of the Gaussian in 
      that direction, :math:`\sigma_b`. 
      
    The algorith proceeds as follows. 
    1. Input all data points (x,y)
    2. Compute the centroid, semi-major and semi-minor axes of the ellipse. 
    3. Transform all the input points to a coordinate system where
       their first coordainte is distance along the semi-major axis in 
       units of :math:`\sigma_a`, and the second coordinate is the distance 
       along the semi-minor axis in units of :math:`\sigma_b`. This
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
    flag = flag.astype(bool)
        
    idx = find_outliers(x, y, initial_clip=flag, threshold=1e-5)
    mu_x = np.mean(x[~idx])
    mu_y = np.mean(y[~idx])
    sma, smi = compute_eigen_vectors(x[~idx], y[~idx])
    


    plt.cla()
    plt.plot(x, y, 'ko', label="Centroids", zorder=+5)
    plt.plot(x[idx], y[idx], 'o', color='pink', label="Outliers", zorder=+6)
    
    #prob = compute_prob_of_points(x, y, sma, smi)
    for i in range(len(x)):
        plt.text(x[i], y[i], " %i" %(i), zorder=+5)
    
                   
    sigma_a = np.linalg.norm(sma)
    sigma_b = np.linalg.norm(smi)
    angle_deg = np.degrees(np.arctan2(sma[1], sma[0]))

    #plt.plot([mu_x, mu_x+sma[0]], [mu_y, mu_y+sma[1]], 'r-')
    #plt.plot([mu_x, mu_x+smi[0]], [mu_y, mu_y+smi[1]], 'b-')
    #idebug()
    
    #print(sigma_a, sigma_b)
    #idebug()
    ax  = plt.gca()
    if 1:
        for p in [.68, .95, .997]:
            scale = spstats.rayleigh().isf(1-p) 
            width = 2 * sigma_a * scale 
            height= 2 * sigma_b * scale  
            ell = Ellipse(
                [mu_x, mu_y], 
                width=width,
                height=height,
                angle=angle_deg, 
                color='gray',
                alpha=.2,
                label="Prob",
            )
            ax.add_patch(ell)
 
        #scale = inverseChiSquare(1-p)  # Convert prob to chisq
        #sma = np.sqrt(scale * eigenVals[0])
        #smi = np.sqrt(scale * eigenVals[1])
        #ell = Ellipse(xy=[muX,muY], width=2 * sma, height=2 * smi,
        #angle=angle_deg, **kwargs)
        #ax.add_patch(ell)
    plt.axhline(0)
    plt.axvline(0)
    plt.plot(0, 0, '*', color='c', ms=28, mec='w', mew=2)
    plt.xlabel("Column shift (pixels)")
    plt.ylabel("Row shift (pixels)")
    plt.axis('equal')
    plt.legend()
    
    msg = "Prob Transit on Target: %.1e" %(compute_probability_of_offset(x, y))
    plt.title(msg)
        

def compute_probability_of_offset(x, y):
    idx = find_outliers(x, y)
    centroid = get_centroid_point(x[~idx], y[~idx])
    sma, smi = compute_eigen_vectors(x[~idx], y[~idx])
    prob = compute_prob_of_points([0],[0], sma, smi, centroid)
    return prob



def find_outliers(x, y, threshold=1e-6, initial_clip=None, max_iter=10):
    """Find data points that are not outliers 
    
    Use a sigma-clipping algorithm to identify outlier points 
    in the distribution of points (x,y), then return the indices 
    of the inliers, i.e the good data points.
    
    This can crash if everything gets clipped!
    """
    
    #idx is true if a point is bad
    idx = initial_clip
    if idx is None:
        idx = np.zeros(len(y), dtype=bool)

    assert len(x) == len(y)
    assert len(idx) == len(y)

    #idebug()
    old_num_clipped = np.sum(idx)
    for i in range(int(max_iter)):
        sma, smi = compute_eigen_vectors(x[~idx], y[~idx])
        prob = compute_prob_of_points(x, y, sma, smi)
        print(prob)
        new_idx = idx | (prob < threshold)
        new_num_clipped = np.sum(new_idx)

        #print( "Iter %i: %i (%i) clipped points " 
            #%(i, new_num_clipped, old_num_clipped))

        if new_num_clipped == old_num_clipped:
            return new_idx

        old_num_clipped = new_num_clipped
        idx = new_idx
        i += 1
        
        
    #Return indices of *good* points
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
    
    #idebug()
    sma_vec = eigenVecs[:, 0] * np.sqrt(eigenVals[0])
    smi_vec = eigenVecs[:, 1] * np.sqrt(eigenVals[1])
    return sma_vec, smi_vec
    
    #centroid_vec = [np.mean(x), np.mean(y)]
    #centroid_vec = np.array(centroid_vec)
    #return centroid_vec, sma_vec, smi_vec
    
    

def compute_prob_of_points(x, y, sma_vec, smi_vec, cent_vec=None):
    """Compute the probability of observing a point as far away as (x,y) for 
    a given ellipse. 
    
    For the ellipse described by centroid, sma and smi, compute the
    probabliity of observing a point at least as far away as x,y in 
    the direction of that point. 
    
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

    xy = np.vstack([x,y]).transpose()
    rel_vec = xy - cent_vec 
    
    #The covector of a vector **v** is defined here as a vector that
    #is parallel to **v**, but has length :math:`= 1/|v|`
    #Multiplying a vector by the covector of the semi-major axis 
    #gives the projected distance of that vector along that axis. 
    sma_covec = sma_vec / np.linalg.norm(sma_vec)**2
    smi_covec = smi_vec / np.linalg.norm(smi_vec)**2
    coeff1 = np.dot(rel_vec, sma_covec)  #Num sigma along major axis
    coeff2 = np.dot(rel_vec, smi_covec)  #Num sigma along minor axis
    
    dist_sigma = np.hypot(coeff1, coeff2)
    prob = spstats.rayleigh().sf(dist_sigma)
    return prob
    

def get_centroid_point(x, y):
    return np.array([np.mean(x), np.mean(y)])
