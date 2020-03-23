# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy import constants

def loadSettings(dataPath=None, elv=90, nfft=512,
                 maxVel=3, minVel=-3, ndgsVal=30, 
                 freq=np.array([9.5e9, 35e9, 95e9]),
                 maxHeight=5500, minHeight=0, heightRes=50):
    
    """
    This function defines the settings for starting the 
    calculation.
    
    Parameters
    ----------
    dataPath: path to the output from McSnow (mandaroty)
    elv: radar elevation (default = 90) [°]
    nfft: number of fourier decomposition (default = 512) 
    maxVel: maximum fall velocity (default = 3) [m/s]
    minVel: minimum fall velocity (default = -3) [m/s]
    ndgsVal: number of division points used to integrate over the particle surface (default = 30)
    freq: radar frequency (default = 9.5e9, 35e9, 95e9) [Hz]
    maxHeight: maximum height (default = 5500) [m]
    minHeight: minimun height (default = 0) [m]
    heightRes: resolution of the height bins (default = 50) [m]
    
    Returns
    -------
    dicSettings: dictionary with all parameters 
    for starting the caculations    
    """
    
    if dataPath != None:
    
        dicSettings = {'dataPath':dataPath,
                       'elv':elv, 
                       'nfft':nfft,
                       'maxVel':maxVel,
                       'minVel':minVel,
                       'velRes':(maxVel - minVel)/nfft,
                       'freq':freq,
                       'wl':(constants.c / freq) * 1e3, #[mm]
                       'ndgsVal':ndgsVal,
                       'maxHeight':maxHeight,
                       'minHeight':minHeight,
                       'heightRes':heightRes,
                       'heightRange':np.arange(minHeight, maxHeight, heightRes)
                       }

        velBins = np.arange(minVel, maxVel, dicSettings['velRes'])
        velCenterBin = velBins[0:-1]+np.diff(velBins)/2.

        dicSettings['velBins']=velBins
        dicSettings['velCenterBin']=velCenterBin
        
    else:
        msg = ('\n').join(['please load the path to the McSnow output', 
                            'use the dataPath parameter for it', 
                            'e.g. loadSettings(dataPath="/data/path/.")'])
        print(msg)
        dicSettings = None
    
    return dicSettings
                 


