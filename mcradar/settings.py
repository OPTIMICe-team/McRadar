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
    #TODO: make SSRGA dependent on aspect ratio, since alpha_eff depents on it and if we have crystals it of course makes a difference there. Also think about having the LUT not sorted by size but rather mass. I could calculate diameter in direction of travel (so vertical extend) from Dmax and ar.
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
                        - DDA -> this mode uses DDA table. Sofar only Dendrite and for X and W-Band. Selection is only based on size, no ar. 
                          We need to think about how to change that in the future  
      scatSet['lutPath']: in case scatSet['mode'] is either 'table' or 'wisdom' or 'SSRGA' or 'SSRGA-Rayleigh' or 'DDA' the path to the lut.nc files is required
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
                lutFiles = glob(scatSet['lutPath']+'LUT*.nc') # TODO: change back to old file name!!
                listFreq = [l.split('LUT_')[-1].split('.nc')[0].split('Hz_')[0] for l in lutFiles]
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('LUT_')[-1].split('.nc')[0].split('Hz_')[-1] for l in lutFiles]
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
        #dicSettings['elv'] = 90 # TODO: once elevation gets flexible, need to change that back
        if (scatSet['mode'] == 'SSRGA'):
            print('with mode SSRGA, no polarimetric output is generated.')
        else:
            print('SSRGA for aggregates and Rayleigh for monomers. No polarimetric output is generated.')
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                if 'particle_name' in scatSet.keys():
                    msg = 'Using LUTs in ' + scatSet['lutPath']
                    lutFile = scatSet['lutPath']+scatSet['particle_name']+'_LUT.nc'
                    print(lutFile)
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
        dicSettings['elv'] = 90 # TODO: once elevation gets flexible, need to change that back
        print('scattering mode Rayleigh for all particles, only advisable for low frequency radars. No polarimetric output is generated. Also: only 90° elevation')
    elif scatSet['mode'] == 'DDA': 
        
        print('you selected DDA as scattering mode. For now the scattering is calculated from a LUT, and the closest scattering point is selected only by choosing the closest size, mass, aspect ratio. Right now only for plate-like crystal and dendritic aggregate. Careful: right now only possible for W-Band, as for the aggregates thats the only one calculated!!')
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                msg = 'Using LUTs in ' + scatSet['lutPath']
                lutFiles = glob(scatSet['lutPath']+'DDA_LUT_dendrites_freq*.nc') 
                listFreq = [l.split('DDA_LUT_dendrites_')[1].split('_elv')[0].split('freq')[1] for l in lutFiles]
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('elv')[1].split('.nc')[0] for l in lutFiles]
                listElev = list(dict.fromkeys(listElev))
                dicSettings['scatSet']['lutFreqMono'] = [float(f) for f in listFreq]
                dicSettings['scatSet']['lutElevMono'] = [int(e) for e in listElev]
                #- now same for aggregates
                lutFiles = glob(scatSet['lutPath']+'DDA_LUT_dendrite_aggregates_freq*.nc') 
                #listFreq = [l.split('LUT_dendrite_aggregates')[1].split('_elv')[0].split('freq')[1] for l in lutFiles]
                listFreq = [l.split('DDA_LUT_dendrite_aggregates_freq')[1].split('_elv')[0] for l in lutFiles]
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('elv')[1].split('.nc')[0] for l in lutFiles]
                listElev = list(dict.fromkeys(listElev))
                dicSettings['scatSet']['lutFreqAgg'] = [float(f) for f in listFreq]
                dicSettings['scatSet']['lutElevAgg'] = [int(e) for e in listElev]
                
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
        
        #if float(dicSettings['freq']) != float(94e9):
        #    msg = ('\n').join(['with this scattering mode ', scatSet['mode'],
        #                       'only freq=94.00GHz is possible!',
        #                       'check your settings'])
        #    dicSettings = None
        print(msg)
    elif scatSet['mode'] == 'DDA_rational': 
        
        print('you selected DDA using rational functions as scattering mode. The scattering is calculated with rational functions where the fitting parameters have been determinded before. For dendrite aggregates I dont have a solution yet!!')
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                msg = 'Using LUTs in ' + scatSet['lutPath']
                lutFiles = glob(scatSet['lutPath']+'fitting_parameters_rationalFunc_freq*.nc') 
                listFreq = [l.split('LUT_dendrites_')[1].split('_elv')[0].split('freq')[1] for l in lutFiles]
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('elv')[1].split('.nc')[0] for l in lutFiles]
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
    elif scatSet['mode'] != 'full':
        print('scatSet[mode] must be either full (default), table or wisdom or SSRGA or Rayleigh or SSRGA-Rayleigh or DDA')
        dicSettings = None

    return dicSettings
                 


