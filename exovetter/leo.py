import numpy as np

def phasefold(t, per, epo):
    # Phase will span -0.5 to 0.5, with transit centred at phase 0
    phase = np.mod(t - epo, per) / per
    phase[phase > 0.5] -= 1
    return phase

def weighted_mean(y, dy):
    w = 1 / dy**2
    mean = np.sum(w * y) / np.sum(w)
    return mean

def weighted_std(y, dy):
    w = 1 / dy**2
    N = len(w)
    mean = np.sum(w * y) / np.sum(w)
    std = np.sqrt(np.sum(w * (y - mean) ** 2) / ((N - 1) * np.sum(w) / N))
    return std

def weighted_err(y, dy):
    w = 1 / dy**2
    err = 1 / np.sqrt(np.sum(w))
    return err

class Leo:
    def __init__(self, time, per, epo, dur, flux, flux_err, frac, max_chases_phase):
        '''
        Parameters
        -----------
        time : array
            Array of times from lc

        per : float
            Orbital period in days

        epo : float
            Time of first transit in TESS BJD

        dur : float
            Transit duration in days

        flux : array
            Array of flux values from lc

        flux_err : array
            Array of flux error values from lc
        
        frac : float
            fraction of SES for a transit which triggers the chases false alarm statistic (default 0.7)

        max_chases_phase : float
            maximum  to allow the chases search to run on (default 0.1)

        Attributes
        ------------
        qtran : Transit duration divided by the period of the transit
        phase : Array of phases for the time series spanning -0.5 to 0.5 with transit at 0
        in_tran : Phases in transit
        near_tran : Boolean of cadences within 1 transit duration
        epochs : Number of transits accounting for gaps
        tran_epochs : Epochs of the transits
        N_transit : Length of tran_epochs
        fit_tran : Cadences within 2 transit durations
        zpt : Out-of-transit wieghted mean of the fluxes
        dep : Depth of the transit based on the weighted mean of the in transit points
        '''
        self.time = time
        self.per = per
        self.epo = epo
        self.dur = dur
        self.flux = flux
        self.flux_err = flux_err
        self.qtran = dur / per
        
        # Phase spans -0.5 to 0.5 with transit at 0
        self.phase = phasefold(time, per, epo)
        # Cadences in-transit
        self.in_tran = abs(self.phase) < 0.5 * self.qtran
        # Cadences within 1 transit duration
        self.near_tran = abs(self.phase) < self.qtran
        # Actual number of transits accounting for gaps
        self.epochs = np.round((time - epo) / per)
        self.tran_epochs = np.unique(self.epochs[self.in_tran])
        self.N_transit = len(self.tran_epochs)
        # Cadences within 2 transit durations
        self.fit_tran = abs(self.phase) < 2 * self.qtran # can change to a variable other than 2
        # Number of transit datapoints
        self.n_in = np.sum(self.in_tran)
        
        # Out-of-transit flux and transit depth
        self.zpt = weighted_mean(self.flux[~self.near_tran], self.flux_err[~self.near_tran])
        self.dep = self.zpt - weighted_mean(self.flux[self.in_tran], self.flux_err[self.in_tran])

        self.frac = frac
        self.max_chases_phase = max_chases_phase


    def get_SES_MES(self):
        N = len(self.time)
        dep_SES = np.zeros(N)
        n_SES = np.zeros(N)
        dep_MES = np.zeros(N)
        n_MES = np.zeros(N)
        N_transit_MES = np.zeros(N)
        bin_flux = np.zeros(N)
        bin_flux_err = np.zeros(N)
        phase = phasefold(self.time, self.per, self.epo)
        phase[phase < 0] += 1
        for i in np.arange(N):
            # Get individual transit depth at this cadence, i.e. only use datapoints close in time
            in_tran = abs(self.time - self.time[i]) < 0.5 * self.dur
            n_SES[i] = np.sum(in_tran)
            dep_SES[i] = self.zpt - weighted_mean(
                self.flux[in_tran], self.flux_err[in_tran]
            )
            # Get overall transit depth at this cadence, i.e. use all datapoints close in phase
            all_tran = (abs(phase - phase[i]) < 0.5 * self.qtran) | (
                abs(phase - phase[i]) > 1 - 0.5 * self.qtran
            )
            n_MES[i] = np.sum(all_tran)
            dep_MES[i] = self.zpt - weighted_mean(
                self.flux[all_tran], self.flux_err[all_tran]
            )
            epochs = np.round((self.time - self.time[i]) / self.per)
            tran_epochs = np.unique(epochs[all_tran])
            N_transit_MES[i] = len(tran_epochs)
            # Get running mean and uncertainty of out-of-transit fluxes, binned over transit timescale
            in_bin = in_tran & ~self.near_tran
            bin_flux[i] = weighted_mean(self.flux[in_bin], self.flux_err[in_bin])
            bin_flux_err[i] = weighted_err(self.flux[in_bin], self.flux_err[in_bin])
        # Estimate white and red noise following Hartman & Bakos (2016)
        mask = ~np.isnan(bin_flux) & ~self.near_tran
        std = weighted_std(self.flux[mask], self.flux_err[mask])
        bin_std = weighted_std(bin_flux[mask], bin_flux_err[mask])
        expected_bin_std = (
            std
            * np.sqrt(np.nanmean(bin_flux_err[mask] ** 2))
            / np.sqrt(np.nanmean(self.flux_err[mask] ** 2))
        )
        self.sig_w = std
        sig_r2 = bin_std**2 - expected_bin_std**2
        self.sig_r = np.sqrt(sig_r2) if sig_r2 > 0 else 0
        # Estimate signal-to-pink-noise following Pont et al. (2006)
        self.err = np.sqrt(
            (self.sig_w**2 / self.n_in) + (self.sig_r**2 / self.N_transit)
        )
        err_SES = np.sqrt((self.sig_w**2 / n_SES) + self.sig_r**2)
        err_MES = np.sqrt((self.sig_w**2 / n_MES) + (self.sig_r**2 / N_transit_MES))
        self.SES_series = dep_SES / err_SES
        self.dep_series = dep_MES
        self.err_series = err_MES
        self.MES_series = dep_MES / err_MES
        self.MES = self.dep / self.err
        Fmin = np.nanmin(-self.dep_series)
        Fmax = np.nanmax(-self.dep_series)
        self.SHP = Fmax / (Fmax - Fmin)

    def get_chases(self):
        deps = np.zeros(self.N_transit)
        errs = np.zeros(self.N_transit)
        self.SES = np.zeros(self.N_transit)
        self.rubble = np.zeros(self.N_transit)
        self.chases = np.zeros(self.N_transit)
        # self.redchi2 = np.zeros(self.N_transit)
        # Search range for chases metric is between 1.5 durations and n times the period away
        chases_tran = (abs(self.phase) > 1.5 * self.qtran) & (abs(self.phase) < self.max_chases_phase)
    
        # Get metrics for each transit event
        for i in range(self.N_transit):
            epoch = self.tran_epochs[i]
            in_epoch = self.in_tran & (self.epochs == epoch)
            # Compute the transit time, depth, and SES for this transit
            transit_time = self.epo + self.per * epoch
            n_in = np.sum(in_epoch)
            dep = self.zpt - weighted_mean(self.flux[in_epoch], self.flux_err[in_epoch])
            err = np.sqrt((self.sig_w**2 / n_in) + self.sig_r**2)
            deps[i], errs[i] = dep, err
            self.SES[i] = dep / err
            # Find the most significant nearby event
            chases_epoch = (chases_tran & (self.epochs == epoch) & (np.abs(self.SES_series) > self.frac * self.SES[i]))

            if np.any(chases_epoch):
                self.chases[i] = np.min(np.abs(self.time[chases_epoch] - transit_time)) / (
                    self.max_chases_phase * self.per
                )
            else:
                self.chases[i] = 1
            # Find how much of the transit falls in gaps
            fit_epoch = self.fit_tran & (self.epochs == epoch)
            n_obs = np.sum(fit_epoch)
            cadence = np.nanmedian(np.diff(self.time[fit_epoch]))
            n_exp = 4 * self.dur / cadence # 4 is used because of the 2 transit duration on either side above in fit_tran
            self.rubble[i] = n_obs / n_exp
            # if ("transit_aic" in tlc.metrics) and ~np.isnan(tlc.metrics["transit_aic"]):
            #     tm = TransitModel(
            #         tlc.metrics["transit_per"],
            #         tlc.metrics["transit_epo"],
            #         tlc.metrics["transit_RpRs"],
            #         tlc.metrics["transit_aRs"],
            #         tlc.metrics["transit_b"],
            #         tlc.metrics["transit_u1"],
            #         tlc.metrics["transit_u2"],
            #         tlc.metrics["transit_zpt"],
            #     )
            #     resid = tm.residual(
            #         tm.params,
            #         tlc.time[fit_epoch],
            #         tlc.flux[fit_epoch],
            #         tlc.flux_err[fit_epoch],
            #     )
            #     chi2 = np.sum(resid**2)
            #     tlc.redchi2[i] = chi2 / (np.sum(fit_epoch) - 6)
            # else:
            #     tlc.redchi2[i] = np.nan
        O = self.SES
        E = self.dep / errs
        chi2 = np.sum((O - E) ** 2 / E)
        self.CHI = self.MES / np.sqrt(chi2 / (self.N_transit - 1))
        self.med_chases = np.nanmedian(self.chases)
        self.mean_chases = np.nanmean(self.chases)
        self.max_SES = np.nanmax(self.SES)
        self.DMM = np.nanmean(deps) / np.nanmedian(deps)

    def plot (self):
        print('Implement plotting')