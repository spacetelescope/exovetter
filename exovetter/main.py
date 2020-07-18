
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


#The LPP vetter is an example of a Vetter class.
class Lpp(Vetter):
    def __init__(self, **kwargs):
        pass

    def run(self, tce, lightcurve):
        #Actual implementation of LPP is called here
        ...

#The odd even test. We can have many such tests
class OddEven(Vetter):
    def __init__(self, **kwargs):
        pass

    def run(self, tce, lightcurve):
        #Actual implementation of LPP is called here
        ...


def vetTce(tce, vetterList):
    """Vet a single TCE with a list of vetters"""

    #Load lightcurve data, possibly using lightKurve package
    lightcurve = loadLightCurve(tce.ticId, sectorList)

    metrics = dict()
    for v in vetterList:
        resDict = v.apply(tce, lightcurve)
        metrics.update(resDict)
    return metrics


