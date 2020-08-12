#https://github.com/jakevdp/lpproj
import numpy as np
from sklearn.neighbors import NearestNeighbors
from lpproj import LocalityPreservingProjection
import copy

def computeLPPTransitMetric(data, mapInfo):
    """
    This function takes a data class with light curve info
    and the mapInfo with information about the mapping to use.
    It then returns a lpp metric value.
    """
    
    binFlux, binPhase=foldBinLightCurve(data, mapInfo.ntrfr, mapInfo.npts)
    
    #plt.figure()
    #plt.plot(binPhase,binFlux,'.--')
    
    
    #Dimensionality Reduction and knn parts
    rawTLpp, transformedTransit = computeRawLPPTransitMetric(binFlux,mapInfo)
    
    #Normalize by Period Dependence
    normTLpp=periodNormalLPPTransitMetric(rawTLpp,np.array([data.period, data.mes]), mapInfo)
    
    return normTLpp,rawTLpp,transformedTransit
    
    

def runningMedian(t,y,dt,runt):
    """
    Take a running median of size dt
    Return values at times given in runt
    """
    newy=np.zeros(len(y))
    newt=np.zeros(len(y))
    
    srt = np.argsort(t)
    newt = t[srt]
    newy = y[srt]

    runy=[]
    for i in range(len(runt)):      
        tmp=[]
        for j in range(len(newt)):     
            if (newt[j] >= (runt[i]-dt)) and (newt[j] <= (runt[i]+dt)):
                tmp.append(newy[j])
                
        if np.isnan(np.nanmedian(np.array(tmp))) :
            runy.append(0)
        else:
            runy.append(np.nanmedian(np.array(tmp)))
    
    return(list(runt),runy)


def foldBinLightCurve (data, ntrfr, npts):
    """
    Fold and bin light curve for input to LPP metric calculation
    
    data contains time, tzero, dur, period, mes and flux (centered around zero)
    
    ntrfr -- number of transit fraction for binning around transit ~1.5
    npts -- number of points in the final binning.
    
    """

    #Create phase light curve
    phaselc =np.mod((data.time-(data.tzero-0.5*data.period))/data.period,1)
    flux=data.flux
    mes=data.mes
    #Determine the fraction of the time the planet transits the star.
    #Insist that ntrfr * transit fraction
    if ~np.isnan(data.dur) & (data.dur >0):
        transit_dur = data.dur
    else:
        transit_dur = 0.2 * data.period/24.
    
    transit_fr=transit_dur/24./data.period
    if (transit_fr * ntrfr) > 0.5 :
        transit_fr = 0.5/ntrfr
        
    #Specify the out of transit (a) and the in transit regions
    binover=1.3
    if mes <= 20:
        binover=-(1/8.0)*mes + 3.8
        
    endfr = .03
    midfr= .11
    a = np.concatenate((np.arange(endfr,.5-midfr,1/npts) , \
                        np.arange((0.5+midfr),(1-endfr),1/npts)), axis=None)
    ovsamp=4.0
    #bstep=(ovsamp*ntrfr*transit_fr)/npts
    b_num=41
    b =np.linspace((0.5-ntrfr*transit_fr),(0.5+ntrfr*transit_fr),b_num)

    #print "length a: %u " % len(a)
    #print "length b: %u" % len(b)
    [runta,runya] = runningMedian(phaselc,flux,binover/npts,a)
    [runtb,runyb] = runningMedian(phaselc,flux,\
                    (binover*ovsamp*ntrfr*transit_fr)/npts,b)

    #Combine the two sets of bins
    runymess=np.array(runya + runyb)
    runtmess = np.array(runta + runtb)

    srt=np.argsort(runtmess)
    runy=runymess[srt]
    runt=runtmess[srt]
    
    #Scale the flux by the depth so everything has the same depth.
    #Catch or dividing by zero is to not scale.
    scale = -1*np.min(runyb)
    if scale != 0:
        scaledFlux=runy/scale
    else:
        scaledFlux=runy
    
    binnedFlux=scaledFlux
    phasebins=runt
    
    return binnedFlux,phasebins


def computeRawLPPTransitMetric(binFlux,mapInfo):
    """
    Perform the matrix transformation with LPP
    Do the knn test to get a raw LPP transit metric number.
    """
    
    Yorig = mapInfo.YmapMapped
    lpp = LocalityPreservingProjection(n_components=mapInfo.n_dim)
    lpp.projection_ = mapInfo.YmapM
    
    #To equate to Matlab LPP methods, we need to remove mean of transform.
    #Check if this is correct, YmapMean is an array that is transit shapped
    normBinFlux = binFlux - mapInfo.YmapMean
    
    #normBinFlux = binFlux - np.mean(binFlux)
    #print(mapInfo.YmapMean)
    #print(normBinFlux)
    
    inputY=lpp.transform(normBinFlux.reshape(1,-1))
    
    knownTransitsY=Yorig[mapInfo.knnGood,:]
    
    dist, ind = knnDistance_fromKnown(knownTransitsY,inputY,mapInfo.knn)
    
    rawLppTrMetric=np.mean(dist)
    
    return rawLppTrMetric, binFlux
    
def knnDistance_fromKnown(knownTransits, new, knn):
    """
    For a group of known transits and a new one.
    Use knn to determine how close the new one is to the known transits
    using knn minkowski p = 3 ()
    Using scipy signal to do this.
    """
    #p=3 sets a minkowski distance of 3. #Check that you really used 3 for matlab.
    nbrs = NearestNeighbors(n_neighbors=int(knn), algorithm='kd_tree', p=2)
    nbrs.fit(knownTransits)
    
    distances,indices = nbrs.kneighbors(new)
    
    
    return distances, indices
    
      
    
def periodNormalLPPTransitMetric(rawTLpp, newPerMes, mapInfo):
    """
    Normalize the rawTransitMetric value by those with the closest period.
    This part removes the period dependence of the metric at short periods.
    Plus it makes a value near one be the threshold between good and bad.
    
    newPerMes is the np.array([period, mes]) of the new sample
    """
    knownTrPeriods=mapInfo.mappedPeriods[mapInfo.knnGood]
    knownTrMes=mapInfo.mappedMes[mapInfo.knnGood]
    knownTrrawLpp=mapInfo.dymeans[mapInfo.knnGood]
    nPercentil=mapInfo.nPercentil
    nPsample=mapInfo.nPsample
    
    #Find the those with the nearest periods  Npsample-nneighbors
    logPeriods=np.log10(knownTrPeriods)
    logMes=np.log10(knownTrMes)
    knownPerMes=np.stack((logPeriods, logMes), axis=-1)

    np.shape(knownPerMes)
    logNew=np.log10(newPerMes).reshape(1,-1)
    #logNew=np.array([np.log10(newPeriod)]).reshape(1,1)

    dist,ind = knnDistance_fromKnown(knownPerMes,logNew,nPsample)
    
    #Find the nthPercentile of the rawLpp of these indicies
    nearPeriodLpp=knownTrrawLpp[ind]
    
    LppNPercentile = np.percentile(nearPeriodLpp,nPercentil)
    
    NormLppTransitMetric=rawTLpp/LppNPercentile
    
    return NormLppTransitMetric
    
    

def lpp_onetransit(tcedata, mapInfo, ntransit):
    """
    Chop down the full time series to one orbital period.
    Then gather the lpp value for that one transit.
    """
    
    startTime=tcedata.time[0]+ntransit*tcedata.period
    endTime=tcedata.time[0]+(ntransit+1)*tcedata.period + 3/24.0  #A few cadences of overlap
    
    want=(tcedata.time>=startTime) & (tcedata.time<=endTime)
    newtime=tcedata.time[want]
    newflux=tcedata.flux[want]
    
    nExpCad=(tcedata.time[-1]-tcedata.time[0])/tcedata.period
    
    if len(newtime>nExpCad*0.75):
        onetransit=copy.deepcopy(tcedata)
        onetransit.time=newtime
        onetransit.flux=newflux
        normTLpp, rawTLpp, transformedTr=computeLPPTransitMetric(onetransit,mapInfo)
    else:
        normTLpp=np.nan
        rawTLpp=np.nan
        
    return normTLpp,rawTLpp


def lpp_averageIndivTransit(tcedata, mapInfo):
    """
    
    Create the loop over individual transits and return 
    array normalized lpp values, mean and std.
    Input TCE object and mapInfo object.
    
    It is unclear that this individual transit approach
    separates out several new false positives.
    It probably would require retuning for low SNR signals.
    
    """    
    length=tcedata.time[-1]-tcedata.time[0]
    ntransits=int(np.floor(length/tcedata.period))
    
    lppNorms=np.ones(ntransits)
    lppRaws=np.ones(ntransits)
    
    #nExpCad=(tcedata.time[-1]-tcedata.time[0])/tcedata.period
    
    for i in range(ntransits):
        lppNorms[i],lppRaws[i] = lpp_onetransit(tcedata,mapInfo,i)
    
    lppMed=np.nanmedian(lppNorms)
    lppStd=np.nanstd(lppNorms)
    
    return lppNorms,lppMed, lppStd, ntransits


#%----------------
class Lppdata():
    
    def __init__(self, tce, lc, lc_name = "flux"):
        #Expected a tce object
        #Expecting a lightkurve object
        #Needs a check lightcurve function
        
        self.check_tce(tce)
        
        self.tzero = tce['tzero']
        self.dur = tce['duration']

        self.time = lc.time
        self.flux = lc.__dict__[lc_name]


    def check_tce(self, tce):
        
        try:
            self.mes = tce['snr']
        except AttributeError:
            print('WARNING: LPP requires a MES or SNR value stored as snr in the tce. Using a value of 10.')
            self.mes = 10.0
            pass
        try:
            self.period = tce['period']
        except:
            print('Exception: Period required for the TCE to run LPP.')
            

#%---------
import scipy.io as spio

class Loadmap(object):
    
    def __init__(self,filename):
        
        self.filename=filename
        
        self.readMatlabBlob(filename)

    
    def readMatlabBlob(self,filename):
        """
        read in matlab blob
        Using the DV trained one.
        """      

        mat=spio.loadmat(filename,matlab_compatible=True)
        
        #Pull out the information we need.
        
        self.n_dim = mat['mapInfoDV']['nDim'][0][0][0][0]
        self.Ymap = mat['mapInfoDV']['Ymap'][0][0][0][0]
        self.YmapMapping = self.Ymap['mapping']
        self.YmapMean = self.YmapMapping['mean'][0][0][0]
        self.YmapM = self.YmapMapping['M'][0][0]
        self.YmapMapped = self.Ymap['mapped']
        self.knn=mat['mapInfoDV']['knn'][0][0][0][0]
        self.knnGood=mat['mapInfoDV']['knnGood'][0][0][:,0]
        self.mappedPeriods=mat['mapInfoDV']['periods'][0][0][0]
        self.mappedMes=mat['mapInfoDV']['mes'][0][0][0]
        self.nPsample=mat['mapInfoDV']['nPsample'][0][0][0][0]  #number to sample
        self.nPercentil=mat['mapInfoDV']['npercentilTM'][0][0][0][0]
        self.dymeans=mat['mapInfoDV']['dymean'][0][0][0]
        self.ntrfr= 2.0
        self.npts=80.0