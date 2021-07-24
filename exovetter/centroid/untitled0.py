#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 21:09:54 2021

@author: smullally
"""

import numpy
import matplotlib
import exovetter as exo
from exovetter import const
from exovetter import utils
import exovetter.vetters as vet
import lightkurve as lk

candidate = "TOI 1013.01"
tce = utils.get_mast_tce(candidate)
lc = lk.search_lightcurve(candidate, sector=33, exptime=120).download()

tpf = lk.search_targetpixelfile(candidate, sector=33, exptime=120).download()
import pdb; pdb.set_trace()
cent = vet.Centroid()

cent.run(tce[0],tpf, plot=True)