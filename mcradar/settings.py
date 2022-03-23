# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from glob import glob
import numpy as np
from scipy import constants

def loadSettings(dataPath=None, elv=90, nfft=512,
                 convolute=True,nave=19,noise_pow=10**(-40/10),
                 eps_diss=1e-6, theta=0.6 , uwind=10.0 , time_int=2.0 ,
                 maxVel=3, minVel=-3, ndgsVal=30, 
                 freq=np.array([9.5e9, 35e9, 95e9]),
                 maxHeight=5500, minHeight=0,
                 heightRes=50, gridBaseArea=1,
                 scatSet={'mode':'full',
                          'safeTmatrix':False}):
    #TODO: make SSRGA dependent on aspect ratio, since alpha_eff depents on it and if we have crystals it of course makes a difference there. Also think about having the LUT not sorted by size but rather mass
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
    convolute: if True, the spectrum will be convoluted with turbulence and random noise will be added (default = True)
    nave: number of spectral averages (default = 19), needed only if convolute == True
    noise_pow: radar noise power [mm^6/m^3] (default = -40 dB), needed only if convolute == True
    eps_diss: eddy dissipation rate, m/s^2, needed only if convolute == True
    theta: beamwidth of radar, in degree (will later be transformed into rad)
    uwind: vertical wind velocity in m/s
    time_int: integration time of radar in sec
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
                        - SSRGA -> the code uses SSRGA LUT generated with snowScatt, this mode does not produce polarimetry and is therefore separate from mode LUT. 
                          This mode calculated SSRGA regardless of monomer number and aspect ratio. You need to specify LUT path and particle_name (see snowScatt for particle name) 
                        - Rayleigh -> as in SSRGA, LUT that were generated using Rayleigh approximation are used. 
                          Also here no polarimetry so far, therefore separate mode from LUT, will change in future?
                          This mode uses Rayleigh for all particles, regardless of monomer number. 
                          Careful: only use Rayleigh with low frequency such as C,S or X-Band. You need to specify LUT path.
                        - SSRGA-Rayleigh --> this mode uses Rayleigh for the single monomer particles and SSRGA for aggregates.  
      scatSet['lutPath']: in case scatSet['mode'] is either 'table' or 'wisdom' or 'SSRGA', 'Rayleigh' or 'SSRGA-Rayleigh' the path to the lut.nc files is required
      scatSet['particle_name']: in case scatSet['mode'] is either 'SSRGA' or 'SSRGA-Rayleigh' the name of the particle to use SSRGA parameters is required. For a list of names see snowScatt. 
                                A few examples: 'vonTerzi_dendrite' 

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
                       'convolute':convolute,
                       'nave':nave,
                       'noise_pow':noise_pow,
                       'eps_diss':eps_diss,
                       'theta':theta,
                       'time_int':time_int,
                       'uwind':uwind,
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
        print(scatSet)
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                msg = 'Using LUTs in ' + scatSet['lutPath']
                lutFiles = glob(scatSet['lutPath']+'testLUT*.nc') # TODO: change back to old file name!!
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
    elif (scatSet['mode'] == 'SSRGA') or (scatSet['mode'] == 'SSRGA-Rayleigh'):
        print(scatSet)
        if (scatSet['mode'] == 'SSRGA'):
            print('with mode SSRGA, no polarimetric output is generated. Sofar, only elevation = 90° is possible.')
        else:
            print('SSRGA for aggregates and Rayleigh for monomers. No polarimetric output is generated. Sofar, only elevation = 90° possible.')
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                if 'particle_name' in scatSet.keys():
                    msg = 'Using LUTs in ' + scatSet['lutPath']
                    lutFile = scatSet['lutPath']+scatSet['particle_name']+'_LUT.nc'
                    dicSettings['scatSet']['lutFile'] = lutFile
                else:
                    msg = ('n').join(['with this scattering mode ', scatSet['mode'],
                                     'you need to define a particle_name, for a list of valid particle names see snowScatt'])
                    dicSettings = None
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
    elif scatSet['mode'] == 'Rayleigh':
        print('scattering mode Rayleigh for all particles, only advisable for low frequency radars. No polarimetric output is generated. Also: only 90° elevation')
    
    elif scatSet['mode'] != 'full':
        print('scatSet[mode] must be either full (default), table or wisdom or SSRGA')
        dicSettings = None

    return dicSettings
                 


