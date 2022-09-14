import numpy as np
import batman
from lmfit import Parameters
from utils import phasefold, get_mean_and_error, get_SNR

class TransitLightCurve:
    def __init__(self, tic, t, r, y, dy, c, per, epo, dur):
        self.tic = tic
        self.t = t
        self.r = r
        self.y = y
        self.dy = dy
        self.c = c
        self.per = per
        self.epo = epo
        self.dur = dur
        self.qtran = dur/per
        # Phase spans -0.5 to 0.5 with transit at 0
        self.phase = phasefold(t, per, epo)
        # In-transit cadences
        self.in_tran = (abs(self.phase) < 0.5*self.qtran)
        # In-transit cadences on the right side
        self.right_tran = (self.phase > 0) & (self.phase < 0.5*self.qtran)
        # In-transit cadences on the left side
        self.left_tran = (self.phase < 0) & (self.phase > -0.5*self.qtran)
        # In-transit cadences for odd and even transits
        phase2 = np.mod(t - epo, 2*per)/per
        phase2[phase2 > 1] -= 2
        self.odd_tran = (abs(phase2) < 0.5*self.qtran)
        self.even_tran = (abs(phase2) > 1 - 0.5*self.qtran)
        # Cadences within 1 transit duration
        self.near_tran = (abs(self.phase) < self.qtran)
        # Cadences within 2 transit durations
        self.fit_tran = (abs(self.phase) < 2*self.qtran)
        # Cadences before transit
        self.before_tran = (self.phase < -0.5*self.qtran) & (self.phase > -1.5*self.qtran)
        # Cadences after transit
        self.after_tran = (self.phase > 0.5*self.qtran) & (self.phase < 1.5*self.qtran)
        # Get actual number of transits
        self.epochs = np.round((t - epo)/per)
        self.tran_epochs = np.unique(self.epochs[self.in_tran])
        self.Nt = len(self.tran_epochs)
        self.nt = np.sum(self.in_tran)
        self.nbefore = np.sum(self.before_tran)
        self.nafter = np.sum(self.after_tran)
        # Compute MES as transit depth divided by uncertainty in depth
        self.zpt, self.zpt_err = get_mean_and_error(self.y[~self.near_tran], self.dy[~self.near_tran])
        self.dep, self.err, self.MES = get_SNR(self.y[self.in_tran], self.dy[self.in_tran], self.zpt, self.zpt_err)

class TransitModel:
    def __init__(self, per=None, epo=None, RpRs=None, aRs=None, b=None, u1=None, u2=None, zpt=None, params=None):
        if params is None:
            params = Parameters()
            params.add("per", value=per, min=0)
            params.add("epo", value=epo, min=0)
            params.add("b", value=b, min=0)
            params.add("delta", value=b-RpRs, max=1)
            params.add("RpRs", expr="b - delta")
            params.add("aRs", value=aRs, min=0)
            params.add("u1", value=u1, vary=False)
            params.add("u2", value=u2, vary=False)
            params.add("zpt", value=zpt, min=0)    
        self.params = params
        
    def model(self, params, t):
        bparams = batman.TransitParams()
        bparams.t0 = params["epo"].value
        bparams.per = params["per"].value
        bparams.rp = params["RpRs"].value
        bparams.a = params["aRs"].value
        bparams.inc = np.arccos(params["b"].value/params["aRs"].value)*180./np.pi
        bparams.ecc = 0.
        bparams.w = 90.
        bparams.u = [params["u1"].value, params["u2"].value]
        bparams.limb_dark = "quadratic"
        m = batman.TransitModel(bparams, t)
        _model = m.light_curve(bparams) - 1 + params["zpt"].value
        return _model

    def residual(self, params, t, y, dy):
        _model = self.model(params, t)
        resid = np.sqrt((y - _model)**2/dy**2)
        return resid

class TrapezoidModel:
    def __init__(self, per=None, epo=None, dep=None, qtran=None, qin=None, zpt=None, params=None):
        if params is None:
            params = Parameters()
            params.add("per", value=per, min=0)
            params.add("epo", value=epo, min=0)
            params.add("dep", value=dep, min=0, max=1)
            params.add("qtran", value=qtran, min=0, max=1)
            params.add("qin", value=qin, min=0, max=0.5)
            params.add("zpt", value=zpt, min=0)
        self.params = params

    def model(self, params, t):
        dep = params["dep"].value
        qtran = params["qtran"].value
        qin = params["qin"].value
        phase = np.abs(phasefold(t, params["per"].value, params["epo"].value))
        transit = np.zeros(len(phase))
        qflat = qtran*(1 - qin*2.)
        transit[phase <= qflat/2.] = -dep
        in_eg = (phase > qflat/2.) & (phase <= qtran/2.)
        transit[in_eg] = -dep + ((dep/((qtran-qflat)/2.))*(phase[in_eg]-qflat/2.))
        _model = transit + params["zpt"].value
        return _model

    def residual(self, params, t, y, dy):
        _model = self.model(params, t)
        resid = np.sqrt((y - _model)**2/dy**2)
        return resid

class LinearModel:
    def __init__(self, zpt=None, slope=None, params=None):
        if params is None:
            params = Parameters()
            params.add("zpt", value=zpt)
            params.add("slope", value=slope)
        self.params = params

    def model(self, params, t):
        _model = params["zpt"].value + params["slope"].value*t
        return _model

    def residual(self, params, t, y, dy):
        _model = self.model(params, t)
        resid = np.sqrt((y - _model)**2/dy**2)
        return resid
