
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

    def apply(self, tce, lightcurve):
        """Actually run the test. Returns a dictionary of metric values"""
        pass

    def plot(self, tce, lightcurve):
        """Optional, generate a diagnostic plot"""
        pass


#The LPP vetter is an example of a Vetter class.
class Lpp(Vetter):
    ...

    def apply(self, tce, lightcurve):
        #Actual implementation of LPP is called here
        ...

#The odd even test. We can have many such tests
class OddEven(Vetter):
    pass


def main(config):
    tceTable = pd.read_csv('tcelist.csv')
    sectorList = np.arange(25)

    #Initialise vetters. The list of vetters can be hardcoded, or
    #maybe loaded from a file.
    vetterList = []
    for v in [Lpp, OddEven]:
        vetterList.append( v(**config) )

    output = []
    for i, tce in tceTable.iterrows():
        #Load lightcurve data, possibly using lightKurve package
        lightcurve = loadLightCurve(tce.ticId, sectorList)

        metrics = dict()
        for v in vetterList:
            resDict = v.apply(tce, lightcurve)
            metrics.update(resDict)

        #metrics is a dictionary of all metrics for a single TCE from all tests
        #output is a list of such dictionaries.
        output.append(metrics)

    df = makeDataFrameFromListOfDicts(output)
    return df
