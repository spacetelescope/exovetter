#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:03:14 2020

@author: smullally
"""

#For testing.
import sys
import exovetter.vetters as vetters
from lightkurve import search_lightcurvefile
from astropy import units as u
from exovetter.tce import TCE
 
sys.path.append('/Users/smullally/Python_Code/exovetter-repos/official/exovetter/exovetter')
sys.path.append('/Users/smullally/Python_Code/exovetter-repos/official/exovetter/')


period = 3.5224991 * u.day
tzero = 54953.6193 + 2400000.5 - 2454833.0
duration = 3.1906 * u.hour
depth = .009537
target_name = "Kepler-8"
event_name = "Kepler-8 b"

tce = TCE(period=period, tzero=tzero, duration=duration,
              target_name=target_name, depth=depth, event_name=event_name)

"""tce = dict()
tce['period'] = 3.5224991 
tce['tzero'] = 54953.6193 + 2400000.5 - 2454833.0 
tce['snr'] = 30.4
tce['duration'] = 3.1906 
tce['target_name'] = 'KIC 6922244' """

#%%
#tpf = lk.search_targetpixelfile('KIC 6922244', quarter=4).download()
#lc = tpf.to_lightcurve(aperture_mask = tpf.pipeline_mask)
#flat = lc.flatten(window_length=81)

lcf = search_lightcurvefile("KIC 6922244", quarter = 4, mission = "Kepler").download()
lc = lcf.SAP_FLUX.remove_nans().remove_outliers()
flat = lc.flatten(window_length=81)

lpp = vetters.Lpp('/Users/smullally/Python_Code/exoplanetvetting/original/dave/lpp/newlpp/map/combMapDR25AugustMapDV_6574.mat',\
                  lc_name='flux')

result = lpp.run(tce.to_dict(), flat)

lpp.plot()

flat.plot()

