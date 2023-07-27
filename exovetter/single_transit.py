"""Module to handle single_transit vetter"""

import numpy as np

from exovetter import utils

def transit_count(time, period_days, epoch, duration_days):
    """Get single transit statistics
    
    Parameters
    ----------
    time : numpy array
        times
    period_days : float
        period in days
    epoch : `~astropy.units.Quantity`
        epoch of the transit in days
    duration_days : float
        duration of transit in days

    Returns
    -------
    Nt : int
        number of transits
    phase :
        blah
    qtran :
        blah
    in_tran : blah
        blah
    tran_epochs : blah
        blah
    epochs :
        blah
    """
    
    qtran = duration_days/period_days
    phase = utils.phasefold(time, period_days, epoch)
    epochs = np.round((time - epoch)/period_days)
    in_tran = (abs(phase) < 0.5*qtran)
    tran_epochs = np.unique(epochs[in_tran])
    Nt = len(tran_epochs)

    return Nt, phase, qtran, in_tran, tran_epochs, epochs

def single_event_metrics(Nt, phase, qtran, in_tran, tran_epochs, epochs, epo, period_days, flux, error, time, duration_days, cadence_len, frac=0.6):
    """Get rubble and chases arrays
    
    Parameters
    ----------

    Returns
    -------

    """

    deps = np.zeros(Nt)
    SES = np.zeros(Nt)
    rubble = np.zeros(Nt)
    chases = np.zeros(Nt)

    # Search range for Chases metric is between 1.5 durations and 0.1 times the period from the transit centre
    chases_near_tran = (abs(phase) > 1.5*qtran) & (abs(phase) < 0.1) 
    #^It's taking the else path for chases so unsure if correct bc of those hardcoded values
    
    rubble_near_tran = (abs(phase) < qtran) #This is what is used in the TransitLightCurve class

    zpt, zpt_err = utils.get_mean_and_error(flux[~rubble_near_tran], error[~rubble_near_tran]) 
    #^using rubble one since it's the same as the one in TransitLightCurve which the original code uses
    
    #Compute SES series
    n = len(time)
    SES_series = np.zeros(n)
    for i in range(n):
        # Get SES for this cadence - only use datapoints close in time
        in_tran_SES = (abs(time - time[i]) < 0.5*duration_days)
        _, _, SES_series[i] = utils.get_SNR(flux[in_tran_SES], error[in_tran_SES], zpt, zpt_err)

    # Compute individual transit metrics
    for i in range(Nt):
        epoch = tran_epochs[i]
        in_epoch = in_tran & (epochs == epoch)
        
        # Compute the transit time, depth, and SES for this transit
        transit_time = epo + period_days*epoch
        deps[i], _, SES[i] = utils.get_SNR(flux[in_epoch], error[in_epoch], zpt, zpt_err)
        
        # Find the most significant nearby event
        near_epoch = chases_near_tran & (epochs == epoch) & (np.abs(SES_series) > frac*SES[i])

        if np.any(near_epoch):
            chases[i] = np.min(np.abs(time[near_epoch] - transit_time))/(0.1*period_days)
        else:
            chases[i] = 1
        # Find how much of the transit falls in gaps
        near_epoch = rubble_near_tran & (epochs == epoch)
        nobs = np.sum(near_epoch)
        #cadence = 30 if tlc.c[near_epoch][0] < 40000 else 10 #MD 2023 Commented out 

        cadence = cadence_len
        
        nexp = 2*duration_days*24*60/cadence
        rubble[i] = nobs/nexp

    return chases, rubble
    
