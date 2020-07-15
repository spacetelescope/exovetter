#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:17:57 2020

@author: smullally
"""


import exovetter as vet


def test_lpp():
    """
    Test reads in an lpp file name, runs the metric without \
    any plots and returns values.  
    It then asseerts that the normal one is greater than 0, which must be true.
    """
    
    tce_filename = "tce0001234567_dvt.fits"
    
    #Populate the tce class by reading in a file
    #At this point a phases array, 
    #and (int) transit_number would get populated.
   
    tce = vet.create_tce(file = tce_filename, ext = 1)
    
    #This populates the LPP metric fields in this class
    #It also populates where the plotting data is stored
    #This function relies on inputs from a precomputed model stored
    #in something called mapInfo. In theory if someone trains up another 
    #model we could input it here.
    tce.run_lpp(mapInfo = None)
    
    norm_TLpp = tce.lpp.norm
    raw_TLpp = tce.lpp.raw
    binned_lpp_transit = tce.lpp.binned_transit
    
    plot_out = tce.lpp.plot(filename="lpp_plot.png")
    
    assert norm_TLpp > 0
    assert norm_TLpp < 10
    assert len(binned_lpp_transit) > 50 
    

    



    