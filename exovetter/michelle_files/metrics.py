import numpy as np

from scipy.special import erfcinv
from exovetter.michelle_files.michelle_utils import phasefold, get_SNR

def get_SES_MES(tlc):
    n = len(tlc.t)
    dep_series = np.zeros(n)
    err_series = np.zeros(n)
    SES_series = np.zeros(n)
    MES_series = np.zeros(n)
    phase = phasefold(tlc.t, tlc.per, tlc.epo)
    phase[phase < 0] += 1
    for i in range(n):
        # Get SES for this cadence - only use datapoints close in time
        in_tran = (abs(tlc.t - tlc.t[i]) < 0.5*tlc.dur)
        _, _, SES_series[i] = get_SNR(tlc.y[in_tran], tlc.dy[in_tran], tlc.zpt, tlc.zpt_err)
        # Get MES for this cadence - use all datapoints close in phase
        in_tran = (abs(phase - phase[i]) < 0.5*tlc.qtran) | (abs(phase - phase[i]) > 1-0.5*tlc.qtran)
        dep_series[i], err_series[i], MES_series[i] = get_SNR(tlc.y[in_tran], tlc.dy[in_tran], tlc.zpt, tlc.zpt_err)
    tlc.dep_series = dep_series
    tlc.err_series = err_series
    tlc.SES_series = SES_series
    tlc.MES_series = MES_series
    
def get_single_events(tlc, frac=0.6):
    deps = np.zeros(tlc.Nt)
    SES = np.zeros(tlc.Nt)
    rubble = np.zeros(tlc.Nt)
    chases = np.zeros(tlc.Nt)
    if not hasattr(tlc, "SES_series"):
        print("Warning! SES time series was not computed. Computing now...")
        get_SES_MES(tlc)
    # Search range for Chases metric is between 1.5 durations and 0.1 times the period from the transit centre
    near_tran = (abs(tlc.phase) > 1.5*tlc.qtran) & (abs(tlc.phase) < 0.1)
    # Compute individual transit metrics
    for i in range(tlc.Nt):
        epoch = tlc.tran_epochs[i]
        in_epoch = tlc.in_tran & (tlc.epochs == epoch)
        # Compute the transit time, depth, and SES for this transit
        transit_time = tlc.epo + tlc.per*epoch
        deps[i], _, SES[i] = get_SNR(tlc.y[in_epoch], tlc.dy[in_epoch], tlc.zpt, tlc.zpt_err)
        # Find the most significant nearby event
        near_epoch = near_tran & (tlc.epochs == epoch) & (np.abs(tlc.SES_series) > frac*SES[i])
        if np.any(near_epoch):
            chases[i] = np.min(np.abs(tlc.t[near_epoch] - transit_time))/(0.1*tlc.per)
        else:
            chases[i] = 1
        # Find how much of the transit falls in gaps
        near_epoch = tlc.near_tran & (tlc.epochs == epoch)
        nobs = np.sum(near_epoch)
        #cadence = 30 if tlc.c[near_epoch][0] < 40000 else 10 #MD 2023 Commented out 

        cadence = tlc.cadence_len
        
        nexp = 2*tlc.dur*24*60/cadence
        rubble[i] = nobs/nexp

    tlc.deps = deps
    tlc.SES = SES
    tlc.chases = chases
    tlc.rubble = rubble

def get_uniqueness(tlc, nTCE=20000):
    if not hasattr(tlc, "MES_series"):
        print("Warning! MES time series was not computed. Computing now...")
        get_SES_MES(tlc)
    if not hasattr(tlc, "SES"):
        print("Warning! Individual transit metrics were not computed. Computing now...")
        get_single_events(tlc)
    tlc.sig_pri, tlc.sig_sec, tlc.sig_ter, tlc.sig_pos = -1, -1, -1, -1
    tlc.phs_pri, tlc.phs_sec, tlc.phs_ter, tlc.phs_pos = -1, -1, -1, -1
    tlc.sig_oe, tlc.sig_lr = -1, -1
    # Get false alarm thresholds
    tlc.FA1 = np.sqrt(2)*erfcinv((tlc.dur/tlc.per) * (1./nTCE))
    tlc.FA2 = np.sqrt(2)*erfcinv((tlc.dur/tlc.per))
    # Get DMM
    mean_depth = np.nanmean(tlc.deps)
    median_depth = np.nanmedian(tlc.deps)
    tlc.DMM = mean_depth/median_depth
    # Get Shape
    Fmin = np.nanmin(-tlc.dep_series)
    Fmax = np.nanmax(-tlc.dep_series)
    tlc.SHP = Fmax/(Fmax - Fmin)
    # Get odd-even significance
    if np.any(tlc.odd_tran) and np.any(tlc.even_tran):
        odd_dep, odd_err, _ = get_SNR(tlc.y[tlc.odd_tran], tlc.dy[tlc.odd_tran], tlc.zpt, tlc.zpt_err)
        even_dep, even_err, _ = get_SNR(tlc.y[tlc.even_tran], tlc.dy[tlc.even_tran], tlc.zpt, tlc.zpt_err)
        tlc.sig_oe = np.abs(odd_dep - even_dep)/np.sqrt(odd_err**2 + even_err**2)
    # Get transit aysmmetry metric
    if np.any(tlc.left_tran) and np.any(tlc.right_tran):
        left_dep, left_err, _ = get_SNR(tlc.y[tlc.left_tran], tlc.dy[tlc.left_tran], tlc.zpt, tlc.zpt_err)
        right_dep, right_err, _ = get_SNR(tlc.y[tlc.right_tran], tlc.dy[tlc.right_tran], tlc.zpt, tlc.zpt_err)
        tlc.sig_lr = np.abs(left_dep - right_dep)/np.sqrt(left_err**2 + right_err**2)
    # Get information from full MES series
    phase = phasefold(tlc.t, tlc.per, tlc.epo)
    phase[phase < 0] += 1
    # Get primary significance
    arg_pri = np.argmax(tlc.MES_series[tlc.in_tran])
    tlc.sig_pri = tlc.MES_series[tlc.in_tran][arg_pri]
    tlc.phs_pri = phase[tlc.in_tran][arg_pri]
    # Get secondary significance - at least 2 transit durations from primary
    mask = (abs(phase - tlc.phs_pri) < 2*tlc.qtran) | (abs(phase - tlc.phs_pri) > 1-2*tlc.qtran)
    if not np.any(~mask):
        return
    arg_sec = np.argmax(tlc.MES_series[~mask])
    tlc.sig_sec = tlc.MES_series[~mask][arg_sec]
    tlc.phs_sec = phase[~mask][arg_sec]
    tlc.dep_sec = tlc.dep_series[~mask][arg_sec]
    tlc.err_sec = tlc.err_series[~mask][arg_sec]
    # Get Fred excluding primary and secondary
    non_pri_sec = ~mask & ~(abs(phase - tlc.phs_sec) < tlc.qtran)
    # Red noise is std of measured amplitudes
    red_noise = np.sqrt(np.cov(tlc.dep_series[non_pri_sec], aweights=1./tlc.err_series[non_pri_sec]**2))
    # White noise is std of photometric data points
    white_noise = np.sqrt(np.cov(tlc.y[non_pri_sec], aweights=1./tlc.dy[non_pri_sec]**2))
    tlc.Fred = np.sqrt(tlc.nt)*red_noise/white_noise   
    # Get tertiary significance - at least 2 transit durations from primary and secondary
    mask = mask | (abs(phase - tlc.phs_sec) < 2*tlc.qtran)
    if not np.any(~mask):
        return
    arg_ter = np.argmax(tlc.MES_series[~mask])
    tlc.sig_ter = tlc.MES_series[~mask][arg_ter]
    tlc.phs_ter = phase[~mask][arg_ter]
    # Get positive significance - at least 3 transit durations from primary and secondary
    mask = (abs(phase - tlc.phs_pri) < 3*tlc.qtran) | (abs(phase - tlc.phs_pri) > 1-3*tlc.qtran) | (abs(phase - tlc.phs_sec) < 3*tlc.qtran)
    if not np.any(~mask):
        return
    arg_pos = np.argmax(-tlc.MES_series[~mask])
    tlc.sig_pos = -tlc.MES_series[~mask][arg_pos]
    tlc.phs_pos = phase[~mask][arg_pos]

def recompute_MES(tlc, chases=0.01, rubble=0.75):
    if not hasattr(tlc, "SES"):
        print("Warning! Individual transit metrics were not computed. Computing now...")
        get_single_events(tlc) 
    rubble_flag = (tlc.rubble <= rubble)
    zuma_flag = (tlc.SES < 0)
    if tlc.Nt <= 5:       
        chases_flag = (tlc.chases < chases)
        bad_epochs = (chases_flag | rubble_flag | zuma_flag)
    else:
        bad_epochs = (rubble_flag | zuma_flag)
    use_tran = (tlc.in_tran & ~np.isin(tlc.epochs, tlc.tran_epochs[bad_epochs]))
    if np.any(use_tran):
        _, _, tlc.new_MES = get_SNR(tlc.y[use_tran], tlc.dy[use_tran], tlc.zpt, tlc.zpt_err)
        tlc.new_Nt = len(tlc.tran_epochs[~bad_epochs])
    else:
        tlc.new_MES = 0
        tlc.new_Nt = 0

def compute_all_metrics(tlc, chases=0.01, rubble=0.75, frac=0.6, nTCE=20000):
    get_SES_MES(tlc)
    get_single_events(tlc, frac=frac)
    get_uniqueness(tlc, nTCE=nTCE)
    recompute_MES(tlc, chases=chases, rubble=rubble)

