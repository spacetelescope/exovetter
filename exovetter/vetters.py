

#Each vetting test is wrapped in a class that follows this
#structure.
class Vetter():
    def __init__(self, **kwargs):
        """kwargs stores the configuration parameters common to all TCEs
        For example, for the Odd-even test, it might specify the signficance
        of the depth difference that causes a TCE to fail
        """
        pass

    def run(self, tce, lightcurve):
        """Actually run the test. Returns a dictionary of metric values"""
        pass

    def plot(self, tce, lightcurve):
        """Optional, generate a diagnostic plot"""
        pass


import lpp
#The LPP vetter is an example of a Vetter class.
class Lpp(Vetter):
    def __init__(self, map_filename, lc_name = "flux", **kwargs):
        """
        Initializes the LPP Vetter Function and loads the LPP Map.
        
        Parameters
        -----------
            map_filename : string
                location of the lpp .mat file
            lc_name : string
                name of the flux array in the lightkurve object. 
                default is 'flux'
        
        Returns: object
            Lpp Vetter Object
        """
        #https://sourceforge.net/p/lpptransitlikemetric/code/HEAD/tree/data/maps/mapQ1Q17DR24-DVMed6084.mat
        if map_filename =="":
            map_filename = "mapQ1Q17DR24-DVMed6084.mat"
            
        self.map_info = lpp.Loadmap(map_filename)
        self.lc_name = lc_name
        self.tce = None
        self.lc = None
        self.plot_data = None

    def run(self, tce, lightcurve):
        """
        Runs the LPP Vetter on the specified TCE and lightcurve
        and returns the value of the metric.
        
        Parameters
        ----------
            tce : dict 
                  Contains period in days, tzero in units of lc time
                  duration in hours, snr estimate
            
            lightcurve : lightkurve object
                 Contains the detrended light curve's time and flux arrays.
        
        Returns
        --------
            raw_lpp : float
                    Raw LPP value
            norm_lpp  : float
                    Lpp value normalized by period and snr
            plot_data : dictionary 
                    The folded, binned transit prior to the 
                    LPP transformation.
        """
         
        self.tce = tce
        self.lc = lightcurve
        self.lpp_data = lpp.Lppdata(self.tce, self.lc, self.lc_name)
        
        self.norm_lpp, self.raw_lpp, self.plot_data = \
                    lpp.compute_lpp_Transitmetric(self.lpp_data, self.map_info)
        
        result = dict()
        result['raw_lpp'] = self.raw_lpp
        result['norm_lpp'] = self.norm_lpp
        result['plot_data'] = self.plot_data
        
        return(result)
    
    def plot(self):
        if self.plot_data is not None:
            target = self.tce['target']
            lpp.plot_lpp_diagnostic(self.plot_data, target, self.norm_lpp)
        else:
            raise Exception('LPP Plot data is empty. Run lpp.run first.')
        
                                                                       
        

#The odd even test. We can have many such tests
class OddEven(Vetter):
    def __init__(self, **kwargs):
        pass

    def run(self, tce, lightcurve):
        #Actual implementation of LPP is called here
        ...



