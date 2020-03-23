# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np


def lin2db(data):
    """
    Convert from linear to logarithm units
    
    Parameter
    ---------
    data: single value or an array
    
    Returns
    -------
    returns the data converted to dB
    """

    
    return 10*np.log10(data)

def db2lin(data):
    """
    Convert from logarithm to linear units
    
    Parameter
    ---------
    
    data: single value or an array
    
    Returns
    -------
    returns the data converted to linear
    """
    
    return 10**(data/10.)
