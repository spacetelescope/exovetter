# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:15:21 2020

Scaffolding to run the exerimental modshift code goes here
Debugging code goes here
@author: fergal
"""

from exovetter.tce import Tce
import exovetter.vetters as vetters
import exovetter.const as const
import astropy.units as u
from pprint import pprint
import lightkurve as lk

import numpy as np

def load():
    lkfile = lk.search_lightcurvefile('KIC 1026032', quarter=6).download()
    data = lkfile.PDCSAP_FLUX
    return data


def mock():
    #Create a TCE
    tce = Tce()
    tce['period'] = 8.4604378 *  u.day
    tce['epoch'] = 133.778 * u.day
    tce['epoch_offset'] = const.bkjd
    tce['depth'] = .07 * const.ppm
    tce['duration'] = 3 * u.hour

    data = load()
    #Condition the data
    data = data.flatten()
    idx = np.isfinite(data.time) & np.isfinite(data.flux)
    data = data[idx]

    vetter = vetters.ModShift()
    metrics = vetter.run(tce, data)
    pprint(metrics)

