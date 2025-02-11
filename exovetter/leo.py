"""Implementation of Michelle Kunimoto's LEO vetter: https://github.com/mkunimoto/LEO-vetter
See equations 15 on of https://arxiv.org/pdf/1605.06811
"""
import numpy as np

__all__ = ['ses_mes']

def phasefold(t, per, epo):
    """
    The light curve phase-folded on the TCE period, covering orbital phase between -0.5 and 0.5 with the transit centred at 0.
    """
    phase = np.mod(t - epo, per) / per
    phase[phase > 0.5] -= 1
    return phase

def weighted_mean(y, dy):
    if np.all(dy == 0):
        return np.mean(y)

    w = 1 / dy**2
    return np.sum(w * y) / np.sum(w)

def weighted_err(y, dy):
    if np.all(dy == 0):
        return 0.0

    w = 1 / dy**2
    err = 1 / np.sqrt(np.sum(w))
    return err

def weighted_std(y, dy):
    if np.all(dy == 0):
        return np.std(y, ddof=1)

    w = 1 / dy**2
    N = len(w)
    mean = np.sum(w * y) / np.sum(w)
    std = np.sqrt(np.sum(w * (y - mean) ** 2) / ((N - 1) * np.sum(w) / N))
    return std

def ses_mes(time, per, epo, dur, flux, flux_err):
    """
    Returns SES, MES, and related values for TESS data

    Parameters
    ----------
    time : type
        lightcurve time values.
    per : type
        orbital period in days.
    epo : type
        time of first transit in TESS BJD.
    dur : type
        transit duration in days.
    flux : type
        lightcurve flux values.
    flux_err : type
        lightcurve flux error values.
    
    """

    # Define constants to use
    qtran = dur / per
    phase = phasefold(time, per, epo)
    # Cadences within 1 transit duration
    near_tran = abs(phase) < qtran
    # Cadences in-transit
    in_tran = abs(phase) < 0.5 * qtran
    # Number of transit datapoints
    n_in = np.sum(in_tran)
    # Actual number of transits accounting for gaps
    epochs = np.round((time - epo) / per)
    tran_epochs = np.unique(epochs[in_tran])
    N_transit = len(tran_epochs)

    zpt = weighted_mean(flux[~near_tran], flux_err[~near_tran])
    dep = zpt - weighted_mean(flux[in_tran], flux_err[in_tran])

    # Calculate SES and MES etc
    N = len(time)
    dep_SES = np.zeros(N)
    n_SES = np.zeros(N)
    dep_MES = np.zeros(N)
    n_MES = np.zeros(N)
    N_transit_MES = np.zeros(N)
    bin_flux = np.zeros(N)
    bin_flux_err = np.zeros(N)
    phase = phasefold(time, per, epo)
    phase[phase < 0] += 1
    for i in np.arange(N):
        # Get individual transit depth at this cadence, i.e. only use datapoints close in time
        in_tran = abs(time - time[i]) < 0.5 * dur
        n_SES[i] = np.sum(in_tran)
        dep_SES[i] = zpt - weighted_mean(
            flux[in_tran], flux_err[in_tran]
        )
        # Get overall transit depth at this cadence, i.e. use all datapoints close in phase
        all_tran = (abs(phase - phase[i]) < 0.5 * qtran) | (
            abs(phase - phase[i]) > 1 - 0.5 * qtran
        )
        n_MES[i] = np.sum(all_tran)
        dep_MES[i] = zpt - weighted_mean(
            flux[all_tran], flux_err[all_tran]
        )
        epochs = np.round((time - time[i]) / per)
        tran_epochs = np.unique(epochs[all_tran])
        N_transit_MES[i] = len(tran_epochs)
        # Get running mean and uncertainty of out-of-transit fluxes, binned over transit timescale
        in_bin = in_tran & ~near_tran
        bin_flux[i] = weighted_mean(flux[in_bin], flux_err[in_bin])
        bin_flux_err[i] = weighted_err(flux[in_bin], flux_err[in_bin])
    # Estimate white and red noise following Hartman & Bakos (2016)
    mask = ~np.isnan(bin_flux) & ~near_tran
    std = weighted_std(flux[mask], flux_err[mask])
    bin_std = weighted_std(bin_flux[mask], bin_flux_err[mask])
    expected_bin_std = ( std* np.sqrt(np.nanmean(bin_flux_err[mask] ** 2))/ np.sqrt(np.nanmean(flux_err[mask] ** 2)) ) # WE NEED THIS CORRECT FOR IF YOU HAVE NO FLUX ERROR! (the expected r.m.s. of the binned light curve if the noise were uncorrelated in time)
    if np.all(flux_err == 0):
        expected_bin_std = bin_std # If you have no errors the expected binned std is the same as the actual binned std

    sig_w = std
    sig_r2 = bin_std**2 - expected_bin_std**2 # Why when I plug in small values for the error are these not close to the the same?: Because bin_flux_err actually takes into account the length of flux_err[in_bin], that gets summed up so it can vary. flux_err will just be that small number since it's not the weighted error 
    sig_r = np.sqrt(sig_r2) if sig_r2 > 0 else 0
    # Estimate signal-to-pink-noise following Pont et al. (2006)
    err = np.sqrt((sig_w**2 / n_in) + (sig_r**2 / N_transit))
    err_SES = np.sqrt((sig_w**2 / n_SES) + sig_r**2)
    err_MES = np.sqrt((sig_w**2 / n_MES) + (sig_r**2 / N_transit_MES))
    SES_series = dep_SES / err_SES
    dep_series = dep_MES
    err_series = err_MES
    
    MES_series = dep_MES / err_MES
    MES = dep / err
    
    Fmin = np.nanmin(-dep_series)
    Fmax = np.nanmax(-dep_series)
    SHP = Fmax / (Fmax - Fmin)

    ses_mes_dict = {'sig_w':sig_w, 'sig_r':sig_r, 'err':err, 'SES_series':SES_series, 
    'dep_series':dep_series, 'err_series':err_series, 'MES_series':MES_series, 'MES':MES, 'SHP':SHP}

    return ses_mes_dict

def chases_rubble(time, per, epo, dur, flux, flux_err, ses_mes_results, frac, max_chases_phase):
    """
    Returns Chases, Rubble, and related values for TESS data

    Parameters
    ----------
    time : type
        lightcurve time values.
    per : type
        orbital period in days.
    epo : type
        time of first transit in TESS BJD.
    dur : type
        transit duration in days.
    flux : type
        lightcurve flux values.
    flux_err : type
        lightcurve flux error values.
    ses_mes_results : dict
        results from ses_mes()
    frac : float
        fraction of SES for a transit which triggers the chases false alarm statistic (default 0.7)
    max_chases_phase : float
        maximum to allow the chases search to run on (default 0.1)
    """
    qtran = dur / per
    # Phase spans -0.5 to 0.5 with transit at 0
    phase = phasefold(time, per, epo)
    # Cadences in-transit
    in_tran = abs(phase) < 0.5 * qtran
    # Cadences within 1 transit duration
    near_tran = abs(phase) < qtran
    # Cadences within 2 transit durations
    fit_tran = abs(phase) < 2 * qtran
    epochs = np.round((time - epo) / per)
    tran_epochs = np.unique(epochs[in_tran])
    N_transit = len(tran_epochs)
    zpt = weighted_mean(flux[~near_tran], flux_err[~near_tran])
    
    # initialize arrays to fill
    deps = np.zeros(N_transit)
    errs = np.zeros(N_transit)
    SES = np.zeros(N_transit)
    rubble = np.zeros(N_transit)
    chases = np.zeros(N_transit)
    # Search range for chases metric is between 1.5 durations and max_chases_phase times the period away
    chases_tran = (abs(phase) > 1.5 * qtran) & (abs(phase) < max_chases_phase) 
    # We were discussing the possibility of chases_tran = (abs(self.phase) > 1.5 * self.qtran) & (abs(self.phase) < self.max_chases_phase)

    # Get metrics for each transit event
    for i in range(N_transit):
        epoch = tran_epochs[i]
        in_epoch = in_tran & (epochs == epoch)
        # Compute the transit time, depth, and SES for this transit
        transit_time = epo + per * epoch
        n_in = np.sum(in_epoch)
        dep = zpt - weighted_mean(flux[in_epoch], flux_err[in_epoch])
        err = np.sqrt((ses_mes_results['sig_w']**2 / n_in) + ses_mes_results['sig_r']**2)
        deps[i], errs[i] = dep, err
        SES[i] = dep / err
        # Find the most significant nearby event
        chases_epoch = (chases_tran & (epochs == epoch) & (np.abs(ses_mes_results['SES_series']) > frac * SES[i]))
        if np.any(chases_epoch):
            chases[i] = np.min(np.abs(time[chases_epoch] - transit_time)) / (max_chases_phase * per)
        else:
            chases[i] = 1
        # Find how much of the transit falls in gaps
        fit_epoch = fit_tran & (epochs == epoch)
        n_obs = np.sum(fit_epoch)
        cadence = np.nanmedian(np.diff(time[fit_epoch]))
        n_exp = 4 * dur / cadence
        rubble[i] = n_obs / n_exp

    # Have to redefine dep since it is used in the loop
    dep = zpt - weighted_mean(flux[in_tran], flux_err[in_tran])

    O = SES
    E = dep / errs
    chi2 = np.sum((O - E) ** 2 / E)
    
    chi = ses_mes_results["MES"] / np.sqrt(chi2 / (N_transit - 1))
    dmm = np.nanmean(deps) / np.nanmedian(deps)

    chases_rubble_dict = {'CHI':chi, 'med_chases':np.nanmedian(chases), 'mean_chases':np.nanmean(chases), 'max_SES': np.nanmax(SES), 'DMM':dmm, 'chases':chases, 'rubble':rubble}
    
    return chases_rubble_dict