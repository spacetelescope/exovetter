
#A sketch of an architecture. For discussion

#Each vetting test is wrapped in a class that follows this
#structure.
class Vetter():
    def __init__(self, **kwargs):
        """kwargs stores the configuration parameters common to all TCEs
        For example, for the Odd-even test, it might specify the signficance
        of the depth difference that causes a TCE to fail
        """
        pass

    def __call__(self, tce, lightcurve):
        return self.run(tce, lightcurve)

    def run(self, tce, lightcurve):
        """Actually run the test. Returns a dictionary of metric values"""
        pass

    def plot(self, tce, lightcurve):
        """Optional, generate a diagnostic plot"""
        pass


import lpp
#The LPP vetter is an example of a Vetter class.
#Init requires
class Lpp(Vetter):
    def __init__(self, map_filename, lc_name = "flux", **kwargs):
        
        #At Init we need some way of retrieving the LPP MAP file.
        #Or people can provide their own location to the map file.
        #map location
        #https://sourceforge.net/p/lpptransitlikemetric/code/HEAD/tree/data/maps/mapQ1Q17DR24-DVMed6084.mat
        if map_filename =="":
            map_filename = "mapQ1Q17DR24-DVMed6084.mat"
            
        self.map_info = lpp.Loadmap(map_filename)
        self.lc_name = lc_name

    def run(self, tce, lightcurve):
        #Actual implementation of LPP is called here
        #data needs to contain time, tzero, dur, period, mes and flux 
        #My understanding is that comes from 
        
        lpp_data = lpp.Lppdata(tce, lightcurve, self.lc_name )
        
        norm_lpp, raw_lpp, transit_lpp = lpp.computeLPPTransitMetric(lpp_data, self.map_info)
        
        result = dict()
        result['raw_lpp'] = raw_lpp
        result['norm_lpp'] = norm_lpp
        result['transit_lpp'] = transit_lpp
        
        return(result)
                                                                       
        

#The odd even test. We can have many such tests
class OddEven(Vetter):
    def __init__(self, **kwargs):
        pass

    def run(self, tce, lightcurve):
        #Actual implementation of LPP is called here
        ...



