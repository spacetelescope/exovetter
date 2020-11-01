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



def mock():
    tce = Tce()
    tce['period'] = 8.4604378 *  u.day
    tce['epoch'] = 133.778 * u.day
    tce['epoch_offset'] = const.bkjd
    tce['depth'] = .07 * const.ppm
    tce['duration'] = 3 * u.hour

    import lightkurve as lk

    data = lk.search_lightcurvefile('KIC 1026032', quarter=6).download()
    vetter = vetters.ModShift()

    metrics = vetter.run(tce, data)
    pprint(metrics)

