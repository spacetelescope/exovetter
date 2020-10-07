"""Module to handle Threshold Crossing Event (TCE)."""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

__all__ = ['TCE']


class TCE:
    """Class to handle Threshold Crossing Event (TCE).

    Parameters
    ----------
    period : `~astropy.units.Quantity`
        Period of the event.

    tzero : float
        Time of the event.

    duration : `~astropy.units.Quantity`
        Duration of the event.

    depth : float
        The relative depth of the event.

    target_name : str
        Name of the target (star) for plotting.

    event_name : str
        Name of the TCE or planet for plotting.

    """
    def __init__(self, period=1 * u.day, tzero=0, duration=1 * u.hour,
                 depth=1, target_name='target name', event_name="event b"):
        # TODO: Use Quantity throughout to avoid unit conversion
        # using magic numbers.
        self.period = period.to_value(unit=u.day)
        self.tzero = tzero
        self.duration = duration.to_value(unit=u.hour)
        self.depth = depth
        self.target_name = target_name
        self.event_name = event_name

    def to_dict(self):
        """Return TCE attributes as a dictionary."""

        return {'period':  self.period,
                'tzero': self.tzero,
                'duration': self.duration,
                'depth': self.depth,
                'target_name': self.target_name,
                'event_name': self.event_name}

    def check(self):
        """Validate period against duration."""
        if self.period < self.duration / 24:
            raise ValueError(f'The period ({self.period}) is shorter than '
                             f'the duration ({self.duration}).')

    
    def get_boxmodel(self, times):
        """Return box model, which is also stored in ``self.model``."""
        
        offset = 0.5  #This offset is not returned to user.
        model = np.ones(len(times))
        
        self.model_times = times
        self.model_phases = np.fmod(times - self.tzero + (offset * self.period), self.period)
        
        phases = self.model_phases 
        
        want = ((phases > offset * self.period - 0.5 * self.duration / 24) &
                (phases <= offset * self.period + 0.5 * self.duration / 24))

        model[want] = -1 * self.depth
        
        #plt.figure()
        #plt.plot(phases, model,'.')
        #plt.title('tce box model')
        
        self.model = model

        return model
