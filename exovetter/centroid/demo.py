from exovetter.centroid.centroid import get_per_transit_diff_centroid
import astropy.io.fits as pyfits 
import kepler.tpf as ktpf 

def tic_307210830_02_03():
    """First TCE on TIC 307210830 in second sector """
    tic = 307210830
    sector = 2

    period_days = 2.25301
    epoch_btjd = 1355.2867
    duration_days = 1.0185/24.

    return main(tic, sector, period_days, epoch_btjd, duration_days)



def main(tic, sector, period_days, epoch_btjd, duration_days):
    
    path = '/home/fergal/data/tess/hlsp_tess-data-alerts_tess_phot_%011i-s%02i_tess_v1_tp.fits'
    path = path %(tic, sector)
    fits, hdr = pyfits.getdata(path, header=True)
    cube = ktpf.getTargetPixelArrayFromFits(fits, hdr)
    cube = cube[:, 3:9, 2:8]

    time = fits['TIME']
    vetting_results = get_per_transit_diff_centroid(
        time, 
        cube, 
        period_days, 
        epoch_btjd, 
        duration_days,
        plot=True
    )

    return vetting_results
