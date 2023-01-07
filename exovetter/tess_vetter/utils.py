import numpy as np

def phasefold(t, per, epo):
    # Phase will span -0.5 to 0.5, with transit centred at phase 0
    phase = np.mod(t - epo, per)/per
    phase[phase > 0.5] -= 1
    return phase

def get_mean_and_error(y, dy):
    avg = np.average(y, weights=1./dy**2)
    err = 1./np.sqrt(np.sum(1./dy**2))
    return avg, err

def get_SNR(y, dy, zpt, zpt_err):
    avg, err = get_mean_and_error(y, dy)
    dep = zpt - avg
    err = np.sqrt(zpt_err**2 + err**2)
    return dep, err, dep/err
