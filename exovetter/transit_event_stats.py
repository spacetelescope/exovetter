"""Module to handle single_transit vetter"""

import numpy as np
from scipy.special import erfcinv

from exovetter import utils

def transit_count(time, period, epoch, duration):
    """Get single transit statistics (adopted from Michelle's get_single_events)
    
    Parameters
    ----------
    time : numpy array
        times
    period : float
        transit period in same units as time
    epoch : `~astropy.units.Quantity`
        epoch of the transit in same units as time
    duration : float
        duration of transit in same units as time

    Returns
    -------
    Nt : int
        number of transits given the time array
    phase : array of floats
        phases from -0.5 to 0.5 with transit centered at 0
    qtran : float
        fraction of period that is in transit
    in_tran : array of floats
        Values of phase which are within half of a transit duration on either side of the transit  
    tran_epochs : array of ints
        Epochs where the time array is in transit given by the input duration
    epochs : array of ints
        Transit epochs the length of the time array
    """

    if not np.all(np.isfinite(time)):
        raise ValueError('time must contain all finite values')

    qtran = duration/period
    phase = utils.compute_phases(time, period, epoch, offset=0.5) -0.5 # This code requires phases from -0.5 to 0.5 with transit centered at 0 
    # OLD WAY TO GET THE SAME THING utils.phasefold(time, period_days, epoch)
    epochs = np.round((time - epoch)/period)
    in_tran = (abs(phase) < 0.5*qtran)
    tran_epochs = np.unique(epochs[in_tran])
    Nt = len(tran_epochs)

    # Keep Nt, epochs. Others compute in single_event_metrics
    return Nt, phase, qtran, in_tran, tran_epochs, epochs

def single_event_metrics(Nt, phase, qtran, in_tran, tran_epochs, epochs, epo, period_days, flux, error, time, duration_days, cadence_len, frac=0.6):
    """Get rubble and chases arrays
    
    Parameters
    ----------

    Returns
    -------

    """

    if not np.all(np.isfinite(flux)):
        raise ValueError('flux must contain all finite values')
    if len(time) != len(flux):
        raise ValueError('time and flux must be of same length')

    deps = np.zeros(Nt)
    SES = np.zeros(Nt)
    rubble = np.zeros(Nt)
    chases = np.zeros(Nt)

    # Search range for Chases metric is between 1.5 durations and 0.1 times the period from the transit centre
    chases_near_tran = (abs(phase) > 1.5*qtran) & (abs(phase) < 0.1) 
    #^It's taking the else path for chases so unsure if correct
    
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
    # Rubble
    for i in range(Nt):
        epoch = tran_epochs[i]
        in_epoch = in_tran & (epochs == epoch)
    
        # Find how much of the transit falls in gaps
        near_epoch = rubble_near_tran & (epochs == epoch)
        nobs = np.sum(near_epoch)
        
        nexp = 2*duration_days*24*60/cadence_len
        rubble[i] = nobs/nexp
    
    # Chases
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

    return chases, rubble, SES, zpt, zpt_err

def snr_metrics(time, period_days, epoch, duration_days, flux, error, nTCE=20000):
    """Get snr metrics
    
    Parameters
    ----------

    Returns
    -------

    """
    
    #compute SES and MES (adopted from Michelle's get_SES_MES)
    qtran = duration_days/period_days
    phase = utils.phasefold(time, period_days, epoch) #Phase used to compute zpt, zpt_err, dep, err, and MES

    near_tran = (abs(phase) < qtran)
    zpt, zpt_err = utils.get_mean_and_error(flux[~near_tran], error[~near_tran])

    in_tran = (abs(phase) < 0.5*qtran)
    dep, err, MES = utils.get_SNR(flux[in_tran], error[in_tran], zpt, zpt_err)
    
    n = len(time)
    dep_series = np.zeros(n)
    err_series = np.zeros(n)
    SES_series = np.zeros(n)
    MES_series = np.zeros(n)
    phase_non_zero = utils.phasefold(time, period_days, epoch) #phase used to compute SES and MES
    phase_non_zero[phase_non_zero < 0] += 1
    
    for i in range(n):
        # Get SES for this cadence - only use datapoints close in time
        in_tran = (abs(time - time[i]) < 0.5*duration_days)
        _, _, SES_series[i] = utils.get_SNR(flux[in_tran], error[in_tran], zpt, zpt_err)
        # Get MES for this cadence - use all datapoints close in phase
        in_tran = (abs(phase_non_zero - phase_non_zero[i]) < 0.5*qtran) | (abs(phase_non_zero - phase_non_zero[i]) > 1-0.5*qtran)
        dep_series[i], err_series[i], MES_series[i] = utils.get_SNR(flux[in_tran], error[in_tran], zpt, zpt_err)

    
    #compute uniqueness tests (adopted from Michelle's get_uniqueness)
    sig_pri, sig_sec, sig_ter, sig_pos = -1, -1, -1, -1
    phs_pri, phs_sec, phs_ter, phs_pos = -1, -1, -1, -1
    sig_oe, sig_lr = -1, -1

    # Get false alarm thresholds
    FA1 = np.sqrt(2)*erfcinv((duration_days/period_days) * (1./nTCE))
    FA2 = np.sqrt(2)*erfcinv((duration_days/period_days))
    
    #MD, need to compute deps here to seperate this use from get_single_events
    epochs = np.round((time - epoch)/period_days)
    in_tran = (abs(phase) < 0.5*qtran)
    tran_epochs = np.unique(epochs[in_tran])
    Nt = len(tran_epochs)
    deps = np.zeros(Nt)
    for i in range(Nt):
        individual_epoch = tran_epochs[i]
        in_epoch = in_tran & (epochs == individual_epoch)
        # Compute the transit time, depth, and SES for this transit
        transit_time = epoch + period_days*individual_epoch
        deps[i], _, _ = utils.get_SNR(flux[in_epoch], error[in_epoch], zpt, zpt_err)
  
    # Get DMM
    mean_depth = np.nanmean(deps)
    median_depth = np.nanmedian(deps)
    DMM = mean_depth/median_depth
    
    # Get Shape
    Fmin = np.nanmin(-dep_series)
    Fmax = np.nanmax(-dep_series)
    SHP = Fmax/(Fmax - Fmin)
    
    # Get odd-even significance
    # Have to calculate In-transit cadences for odd and even transits to use this outside of Michelle's TransitLightCurve
    phase2 = np.mod(time - epoch, 2*period_days)/period_days
    phase2[phase2 > 1] -= 2
    odd_tran = (abs(phase2) < 0.5*qtran)
    even_tran = (abs(phase2) > 1 - 0.5*qtran)

    if np.any(odd_tran) and np.any(even_tran):
        odd_dep, odd_err, _ = utils.get_SNR(flux[odd_tran], error[odd_tran], zpt, zpt_err)
        even_dep, even_err, _ = utils.get_SNR(flux[even_tran], error[even_tran], zpt, zpt_err)
        sig_oe = np.abs(odd_dep - even_dep)/np.sqrt(odd_err**2 + even_err**2)
    
    # Get transit aysmmetry metric
    # Have to calculate left and right transits here too
    left_tran = (phase < 0) & (phase > -0.5*qtran)
    right_tran = (phase > 0) & (phase < 0.5*qtran)

    if np.any(left_tran) and np.any(right_tran):
        left_dep, left_err, _ = utils.get_SNR(flux[left_tran], error[left_tran], zpt, zpt_err)
        right_dep, right_err, _ = utils.get_SNR(flux[right_tran], error[right_tran], zpt, zpt_err)
        sig_lr = np.abs(left_dep - right_dep)/np.sqrt(left_err**2 + right_err**2)
    
    # Get information from full MES series
    #Already computed this as phase_non_zero, so all 'phase' from here on is phase_non_zero
    # phase = utils.phasefold(time, period_days, epoch) 
    # phase[phase < 0] += 1

    # Get primary significance
    arg_pri = np.argmax(MES_series[in_tran])
    sig_pri = MES_series[in_tran][arg_pri]
    phs_pri = phase_non_zero[in_tran][arg_pri]
    
    # Get secondary significance - at least 2 transit durations from primary
    mask = (abs(phase_non_zero - phs_pri) < 2*qtran) | (abs(phase_non_zero - phs_pri) > 1-2*qtran)
    if not np.any(~mask):
        print('SNR metrics: could not get tertiary significance, no uni_sig_sec, uni_phs_sec, uni_sig_ter, uni_phs_ter, or uni_phs_pos results')
        results_dict = {
        "SES_array": SES_series,
        "MES_array": MES_series,
        "dep_array": dep_series,
        "err_array": err_series,
        "(POTENTIALLY KEEP THIS) MES": MES,
        "transit_depth": dep,
        "(POTENTIALLY KEEP THIS) err": err,
        "uni_sig_pri": sig_pri,
        "uni_sig_sec": sig_sec,
        "uni_sig_ter": sig_ter,
        "uni_oe_dep": sig_oe,
        "uni_mean_med": DMM,
        "uni_shape": SHP,
        "uni_Fred": Fred,
        "uni_sig_FA1": FA1,
        "uni_sig_FA2": FA2,
        "uni_phs_pri": phs_pri,
        "uni_phs_sec": phs_sec,
        "uni_phs_ter": phs_ter,
        "uni_phs_pos": phs_pos
        }
        return results_dict
    arg_sec = np.argmax(MES_series[~mask])
    sig_sec = MES_series[~mask][arg_sec]
    phs_sec = phase_non_zero[~mask][arg_sec]
    dep_sec = dep_series[~mask][arg_sec]
    err_sec = err_series[~mask][arg_sec]
    
    # Get Fred excluding primary and secondary
    non_pri_sec = ~mask & ~(abs(phase_non_zero - phs_sec) < qtran)
    # Red noise is std of measured amplitudes
    red_noise = np.sqrt(np.cov(dep_series[non_pri_sec], aweights=1./err_series[non_pri_sec]**2))
    # White noise is std of photometric data points
    white_noise = np.sqrt(np.cov(flux[non_pri_sec], aweights=1./error[non_pri_sec]**2))
    nt = np.sum(in_tran) #MD had to recalculate nt
    Fred = np.sqrt(nt)*red_noise/white_noise   
    
    # Get tertiary significance - at least 2 transit durations from primary and secondary
    mask = mask | (abs(phase_non_zero - phs_sec) < 2*qtran)
    if not np.any(~mask):
        print('SNR metrics: could not get tertiary significance, no uni_sig_ter, uni_phs_ter, or uni_phs_pos results') 
        results_dict = {
        "SES_array": SES_series,
        "MES_array": MES_series,
        "dep_array": dep_series,
        "err_array": err_series,
        "(POTENTIALLY KEEP THIS) MES": MES,
        "transit_depth": dep,
        "(POTENTIALLY KEEP THIS) err": err,
        "uni_sig_pri": sig_pri,
        "uni_sig_sec": sig_sec,
        "uni_sig_ter": sig_ter,
        "uni_oe_dep": sig_oe,
        "uni_mean_med": DMM,
        "uni_shape": SHP,
        "uni_Fred": Fred,
        "uni_sig_FA1": FA1,
        "uni_sig_FA2": FA2,
        "uni_phs_pri": phs_pri,
        "uni_phs_sec": phs_sec,
        "uni_phs_ter": phs_ter,
        "uni_phs_pos": phs_pos
        }
        return results_dict
    arg_ter = np.argmax(MES_series[~mask])
    sig_ter = MES_series[~mask][arg_ter]
    phs_ter = phase_non_zero[~mask][arg_ter]
    
    # Get positive significance - at least 3 transit durations from primary and secondary
    mask = (abs(phase_non_zero - phs_pri) < 3*qtran) | (abs(phase_non_zero - phs_pri) > 1-3*qtran) | (abs(phase_non_zero - phs_sec) < 3*qtran)
    if not np.any(~mask):
        print('SNR metrics: could not get positive significance, no uni_phs_pos result') 
        results_dict = {
        "SES_array": SES_series,
        "MES_array": MES_series,
        "dep_array": dep_series,
        "err_array": err_series,
        "(POTENTIALLY KEEP THIS) MES": MES,
        "transit_depth": dep,
        "(POTENTIALLY KEEP THIS) err": err,
        "uni_sig_pri": sig_pri,
        "uni_sig_sec": sig_sec,
        "uni_sig_ter": sig_ter,
        "uni_oe_dep": sig_oe,
        "uni_mean_med": DMM,
        "uni_shape": SHP,
        "uni_Fred": Fred,
        "uni_sig_FA1": FA1,
        "uni_sig_FA2": FA2,
        "uni_phs_pri": phs_pri,
        "uni_phs_sec": phs_sec,
        "uni_phs_ter": phs_ter,
        "uni_phs_pos": phs_pos
        }
        return results_dict
    arg_pos = np.argmax(-MES_series[~mask])
    sig_pos = -MES_series[~mask][arg_pos]
    phs_pos = phase_non_zero[~mask][arg_pos]

    results_dict = {
        "SES_array": SES_series,
        "MES_array": MES_series,
        "dep_array": dep_series,
        "err_array": err_series,
        "(POTENTIALLY KEEP THIS) MES": MES,
        "transit_depth": dep,
        "(POTENTIALLY KEEP THIS) err": err,
        "uni_sig_pri": sig_pri,
        "uni_sig_sec": sig_sec,
        "uni_sig_ter": sig_ter,
        "uni_oe_dep": sig_oe,
        "uni_mean_med": DMM,
        "uni_shape": SHP,
        "uni_Fred": Fred,
        "uni_sig_FA1": FA1,
        "uni_sig_FA2": FA2,
        "uni_phs_pri": phs_pri,
        "uni_phs_sec": phs_sec,
        "uni_phs_ter": phs_ter,
        "uni_phs_pos": phs_pos
        }
    
    return results_dict



                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
    
