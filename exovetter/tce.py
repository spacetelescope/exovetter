"""Module to handle Threshold Crossing Event (TCE)."""

import numpy as np
from astropy import units as u

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

        self.check()

    def check(self):
        """Validate period against duration."""
        if self.period < self.duration / 24:
            raise ValueError(f'The period ({self.period}) is shorter than '
                             f'the duration ({self.duration}).')

    def get_boxmodel(self, times):
        """Return box model, which is also stored in ``self.model``."""
        model = np.ones(len(times))

        self.model_times = times
        self.mmodel_phases = phases = np.mod(
            (times - (self.tzero - 0.5 * self.period)) / self.period, 1)

        want = ((phases > -1 * 0.5 * self.duration / 24) &
                (phases < 0.5 * self.duration / 24))

        model[want] = self.depth
        self.model = model

        return model
