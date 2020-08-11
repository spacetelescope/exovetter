#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:12:25 2020

@author: smullally
"""


import exovetter as exo
import lightkurve as lk
import astropyfrom astropy import units as u
from astropy.time import Time


def test_one_lpp():
    #Use Case is get values for one TCE.
    period = 1.235 * u.d
    tzero =  24541250.235 * t.jd
    duration = 5.2 * u.h
    target_name = "TIC 12345678"

    
    tce = exo.TCE(period = period, tzero = tzero, duration = duration
                  target_name = target_name)
    
    #Specify the lightcurve to vet
    mission = "TESS"
    sector = 14
    
    #Generic  function that runs lightkurve and returns a lightkurve object.
    lc = exo.fetch_TESS_lightcurve(target_name, mission=mission, sector = sector)
    
    lpp = exo.Lpp(ddir = "/path/to/file/", lc = "DETRENDED")
    
    result = lpp.apply(tce,lc)
    
    assert result['lpp_lpp'] > 0
    assert result['lpp_normlpp'] > 0
    assert len(result['lpp_binned']) > 10
    

def test_run_many_tces():
    
    #Each element of vet_list needs to contain the TCE info
    #and info to create the ligth curve (filename, or lightkurve call)
    #In this case we likely will have already created the light curves and
    #saved them some place. So this is just a list of filenames.
    vet_list = exo.load_vet_list('filename_tce.csv')
    
    Lpp = exo.lpp('path/to/file', lc = "DETRENDED")
    sap_Mod = exo.modshift(model = "trap", lc = "SAP_FLUX")
    pdc_Mod = exo.modshift(model = "trap", lc = "PDC_FLUX")
    Snr = exo.snr()
    
    vetter_list = [Lpp, sap_Mod, pdc_Mod, Snr]
    
    #Clearly all this below could also be a method some day.
    results = list()
    
    for vet in tce_list:
        #Details of input are hidden here.
        #Likely the lc information points to a filename (s3 buckeet) to load data
        #And put data into the proper format
        tce, lc = exo.load_tce_and_lightcurve(vet)
        
        tce_results = dict()
        
        tce_results.update(tce)
        
        for v in vetter_list:
            
            tce_results.update(v.run(tce, lc))
    
        results.append(tce_results)
    
    assert len(results) == len(vet_list)
    
        
#----
def test_fergal_approach():
    
    tcelist = load_tce_list(filename)
    
    vetterlist = [exo.Lpp('path/to/file')]
    for tce in tcelits:
        metrics = run_many_vetters(tce, vetterlist, lctype="sap", detrend=False)
    
    
def run_many_vetters_tes(tce, vetterlist, kwargs):
    lc = lightkurve.load_lightcurve_tess(tce, **kwargs)
    
    metrics = dict()
    for v in vetterlist:
        metrics.update( v.run(tce, lc))
        
    return metrics
        