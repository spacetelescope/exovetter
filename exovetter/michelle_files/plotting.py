#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:46:05 2022

@author: smullally
"""

import matplotlib.pyplot as plt

def tlc_plot(tlc):
    """
    Creates a standardized plot.    

    Parameters
    ----------
    tlc : tce+lc object
        Created in classes.TransitLightCurve
        
    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    
    ax[0,0].plot(tlc.phase, tlc.y, "k.")
    ax[0,0].plot(tlc.phase[tlc.fit_tran], tlc.y[tlc.fit_tran], "C0.", label="Within 2 transit durations of centre")
    ax[0,0].plot(tlc.phase[tlc.near_tran], tlc.y[tlc.near_tran], "C1.", label="Within 1 transit duration of centre")
    ax[0,0].plot(tlc.phase[tlc.in_tran], tlc.y[tlc.in_tran], "C2.", label="Within 0.5 transit durations of centre")
    ax[0,0].set_xlim([-2*tlc.qtran, 2*tlc.qtran])
    ax[0,0].legend()
    
    ax[0,1].plot(tlc.t, tlc.y, "k.")
    ax[0,1].plot(tlc.t[tlc.odd_tran], tlc.y[tlc.odd_tran], "C0.", label="Odd transits")
    ax[0,1].plot(tlc.t[tlc.even_tran], tlc.y[tlc.even_tran], "C1.", label="Even transits")
    ax[0,1].legend()
    
    ax[1,0].plot(tlc.phase, tlc.y, "k.")
    ax[1,0].plot(tlc.phase[tlc.left_tran], tlc.y[tlc.left_tran], "C0.", label="Left side of transits")
    ax[1,0].plot(tlc.phase[tlc.right_tran], tlc.y[tlc.right_tran], "C1.", label="Right side of transits")
    ax[1,0].set_xlim([-2*tlc.qtran, 2*tlc.qtran])
    ax[1,0].legend()
    
    ax[1,1].plot(tlc.phase, tlc.y, "k.")
    ax[1,1].plot(tlc.phase[tlc.odd_tran], tlc.y[tlc.odd_tran], "C0.", label="Odd transits")
    ax[1,1].plot(tlc.phase[tlc.even_tran], tlc.y[tlc.even_tran], "C1.", label="Even transits")
    ax[1,1].set_xlim([-2*tlc.qtran, 2*tlc.qtran])
    ax[1,1].legend()


def mes_plot(tlc):
    """
    Create a MES_Series plot centered on the transit.
    Plot a histogram of the transit depths series.

    Parameters
    ----------
    tlc : tce-lightcurve object
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(8,5))
    
    plt.plot(tlc.phase, tlc.MES_series, "k.")
    plt.xlim([-2*tlc.qtran, 2*tlc.qtran])
    plt.xlabel("Orbital phase", fontsize=12)
    plt.ylabel("MES", fontsize=12)
    plt.title("MES Series Plot")
    
    plt.figure(figsize=(8,5))
    plt.hist(tlc.deps*1e6, bins=20, histtype="step")
    plt.axvline(x=tlc.dep*1e6, ls="--", alpha=0.5, label="Avg. Depth of Transit")
    plt.xlabel("Transit depth (ppm)", fontsize=12)
    plt.ylabel("Counts", fontsize=12)
    plt.legend()