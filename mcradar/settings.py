# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from glob import glob
import numpy as np
from scipy import constants

def loadSettings(dataPath=None, elv=90, nfft=512,
                 maxVel=3, minVel=-3, ndgsVal=30, 
                 freq=np.array([9.5e9, 35e9, 95e9]),
                 maxHeight=5500, minHeight=0,
                 heightRes=50, gridBaseArea=1,
                 scatSet={'mode':'full',
                          'safeTmatrix':False,}):
    
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
    gridBaseArea: area of the grid base (default = 1) [m^2]
    scatSet: dictionary that defines the settings for the scattering calculations
      scatSet['mode']: string that defines the scattering mode. Valid values are
                        - full -> pytmatrix calculations for each superparticle
                        - table -> use only the LUT values, very fast, skips nan values in LUT
                        - wisdom -> compute the pytmatrix solution where LUT is still nan and update LUT values
      scatSet['lutPath']: in case scatSet['mode'] is either 'table' or 'wisdom' the path to the lut.nc files is required

    Returns
    -------
    dicSettings: dictionary with all parameters
    for starting the caculations
    """

    if 'mode' not in scatSet.keys():
      scatSet['mode'] = 'full'
    if 'safeTmatrix' not in scatSet.keys():
      scatSet['safeTmatrix'] = False

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
                       'heightRange':np.arange(minHeight, maxHeight, heightRes),
                       'gridBaseArea':gridBaseArea,
                       'scatSet':scatSet,
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
    if (scatSet['mode'] == 'table') or (scatSet['mode']=='wisdom'):
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                msg = 'Using LUTs in ' + scatSet['lutPath']
                lutFiles = glob(scatSet['lutPath']+'testLUT*.nc')
                listFreq = [l.split('testLUT_')[-1].split('.nc')[0].split('Hz_')[0] for l in lutFiles]
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('testLUT_')[-1].split('.nc')[0].split('Hz_')[-1] for l in lutFiles]
                listElev = list(dict.fromkeys(listElev))
                dicSettings['scatSet']['lutFreq'] = [float(f) for f in listFreq]
                dicSettings['scatSet']['lutElev'] = [int(e) for e in listElev]

            else:
                msg = ('\n').join(['with this scattering mode ', scatSet['mode'],
                                   'a valid path to the scattering LUT is required',
                                   scatSet['lutPath'], 'is not valid, check your settings'])
                dicSettings = None
        else:
            msg = ('\n').join(['with this scattering mode ', scatSet['mode'],
                               'a valid path to the scattering LUT is required',
                               'check your settings'])
            dicSettings = None
        print(msg)
    elif scatSet['mode'] != 'full':
        print('scatSet[mode] must be either full (default), table or wisdom')
        dicSettings = None

    return dicSettings
                 


