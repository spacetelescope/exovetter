# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:35:53 2020

@author: fergal
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import exovetter.vetters as vetters

class DefaultVetter(vetters.BaseVetter):

    def run(self, tce, lightcurve):
        pass


class ModifiedVetter(vetters.BaseVetter):
    def __init__(self, **kwargs):
        pass

    def run(self, tce, lightcurve):
        pass

def test_string_dunder():
    """Test that the vetter's string method behaves as expected.

    Note: We may choose to improve the string representation at some point
    """

    v = DefaultVetter()

    #No metrics gets returned as an empty dictionary
    assert str(v) == '{}', str(v)

    #A metrics dictionary gets returned as a pprinted string
    v.metrics = dict(key='value')
    assert str(v) == "{'key': 'value'}", str(v)

    w = ModifiedVetter()
    expected = "<test_vetter.ModifiedVetter"
    assert str(w)[:len(expected)] == expected, str(w)


