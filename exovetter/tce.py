#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:42:41 2020

@author: smullally
"""
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

class TCE():
    def __init__(self, period = 1 * u.day, tzero = 0, duration = 1 * u.hour, \
                 depth = 1, target_name = 'target name', event_name = "event b"):
        """
        

        Parameters
        ----------
        period : astropy quantity with unit
            period of the event. The default is 1 * u.day.
        tzero : float optional
            time of the event. The default is 0.
        duration : astropy quantity, 
            duration of the event. The default is 1 * u.hour.
        depth : float, optional
            of the relative depth of the event, The default is 1.
        target_name : string, optional
            Star Name. Provide the name of the target. Mostly used for plots.
        event_name : string, optional
            Name of the TCE or planet. Mostly used for plots.

        Returns
        -------
        None.

        """
 
        self.period = period.to_value(unit = u.day)
        self.tzero = tzero
        self.duration = duration.to_value(unit = u.hour)
        self.depth = depth
        self.target_name = target_name
        self.event_name = event_name
        
        return
        
        
    def to_dict(self):
        
        tce = dict()
        
        tce['period'] = self.period
        tce['tzero'] = self.tzero
        tce['duration'] = self.duration
        tce['depth'] =self.depth
        tce['target_name'] = self.target_name
        tce['event_name'] = self.event_name
        
        return tce
        
    
    def check(self):
        
        if self.period < self.duration / 24:
            print('Error. The period is shorter than the duration.')
        
    
    def get_boxmodel(self, times):
        
        model = np.ones(len(time))
        
        self.model_times = times
        self.mmodel_phases = phaselc =np.mod((time-(self.tzero-0.5*self.period))/self.period,1)
        
        want = (phases > -1 * 0.5 * self.duration/24) & (phases < 0.5 * self.duration/24)
        
        model[want] = self.depth
        self.model = model
        
        return model
    