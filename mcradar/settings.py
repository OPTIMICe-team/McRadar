# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from glob import glob
import numpy as np
from scipy import constants
import time
import pandas as pd
import xarray as xr

def loadSettings(PSD=False,dataPath=None,atmoFile=None, elv=90, nfft=512,
                 convolute=True,nave=19,noise_pow=10**(-40/10),
                 theta=np.array([1.0,0.6,0.6]) , time_int=2.0 , tau=143*1e-9 ,
                 uwind=10.0, eps_diss=np.array([1e-6]), k_theta=np.array([0]),k_phi=np.array([0]),k_r=np.array([0]),shear_height0=0,shear_height1=0,
                 maxVel=3, minVel=-3,  
                 freq=np.array([9.5e9, 35e9, 95e9]),
                 maxHeight=5500, minHeight=0,
                 heightRes=50, gridBaseArea=1,
                 ndgsVal=30,attenuation=False,
                 scatSet={'mode':'full',
                          'safeTmatrix':False}):
    
   # TODO make eps_diss, wind and shear dependent on height (so an array with length of height). One idea: read in a file with height and eps_diss and then it can select the according eps_diss in full_radar that corresponds to the height. Or: already have eps_diss specified for all heights and just loop through it in fullRadar
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
    convolute: if True, the spectrum will be convoluted with turbulence and random noise (default = True)
    nave: number of spectral averages (default = 19), needed only if convolute == True
    noise_pow: radar noise power [mm^6/m^3] (default = -40 dB), needed only if convolute == True
    theta: beamwidth of radar, in degree (will later be transformed into rad)
    time_int: integration time of radar in sec, needed only if convolute == True
    tau: pulse width of radar in seconds (default: 143ns, which is the one used in Ka-Band radar), needed only if convolute == True
    uwind: x component of wind velocity in m/s (horizontal wind), needed only if convolute == True
    eps_diss: eddy dissipation rate, m/s^2, needed only if convolute == True
    k_theta: wind shear in theta direction (when looking zenith this is in x direction). Needs to be provided with shear_height
    k_phi: wind shear in phi direction (when looking zenith this is in y direction) Needs to be provided with shear_height
    k_r: wind shear in r direction (when looking zenith this is in z direction) Needs to be provided with shear_height
    shear_height0: bottom height of wind shear zone
    shear_height1: top height of wind shear zone
    ndgsVal: number of division points used to integrate over the particle surface (default = 30)
    freq: radar frequency (default = 9.5e9, 35e9, 95e9) [Hz]
    maxHeight: maximum height (default = 5500) [m]
    minHeight: minimun height (default = 0) [m]
    heightRes: resolution of the height bins (default = 50) [m]
    gridBaseArea: area of the grid base (default = 1) [m^2]
    attenuation: get path integrated attenuation based on water vapour, O2 and H2O. This uses PAMTRA
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
                        - SSRGA-Rayleigh -> this mode uses Rayleigh for the single monomer particles and SSRGA for aggregates.
                        - DDA -> this mode uses DDA table. Possible frequencies: 9.6GHz, 35.5GHz, 94.0GHz. Selection is based on mass, ar and size. 
                        		 Columnar or plate-like scattering table will be chosen depending on the aspect ratio of the particles. You need to specify the path to the LUT.  
                          
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
        #if len(eps_diss)>1:
        #print(len(eps_diss))
        #quit()
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
                       'tau':tau,
                       'k_theta':k_theta,
                       'k_phi':k_phi,
                       'k_r':k_r,
                       'shear_height0':shear_height0,
                       'shear_height1':shear_height1,
                       'attenuation':attenuation,
                       }

        velBins = np.arange(minVel, maxVel, dicSettings['velRes'])
        velCenterBin = velBins[0:-1]+np.diff(velBins)/2.

        dicSettings['velBins']=velBins
        dicSettings['velCenterBin']=velCenterBin
    elif PSD == True:
        dicSettings = {'elv':elv,
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
                       'tau':tau,
                       'k_theta':k_theta,
                       'k_phi':k_phi,
                       'k_r':k_r,
                       'shear_height0':shear_height0,
                       'shear_height1':shear_height1,
                       'attenuation':attenuation,
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
    print(attenuation)
    if attenuation == True:
        if atmoFile != None: 
            atmoFile = np.loadtxt(atmoFile)
            height = atmoFile[:,0]
            temp = atmoFile[:,2]# -273.15
            atmoPD = pd.DataFrame(data=temp,index=height,columns=['temp'])
            atmoPD.index.name='range'
            atmoPD['press'] = atmoFile[:,3]
            atmoPD['relHum'] = atmoFile[:,6]
            atmoXR = atmoPD.to_xarray()
            atmoReindex = atmoXR.reindex({'range':dicSettings['heightRange']+dicSettings['heightRes']/2},method='nearest')
            dicSettings['temp']=atmoReindex.temp
            dicSettings['relHum']=atmoReindex.relHum
            dicSettings['press']=atmoReindex.press
            
        else:
            msg = ('\n').join(['since you want to do the attenuation correction you need to give an atmoFile as input.'])
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
        #dicSettings['elv'] = 90 # TODO: once elevation gets flexible, need to change that back
        print('scattering mode Rayleigh for all particles, only advisable for low frequency radars. No polarimetric output is generated')
    elif scatSet['mode'] == 'DDA': 
        t0 = time.time()
        print('you selected DDA as scattering mode. The scattering is calculated from a LUT, and the closest scattering point is selected by choosing the closest size, mass, aspect ratio.')
        if 'lutPath' in scatSet.keys():
            if os.path.exists(scatSet['lutPath']):
                msg = 'Using LUTs in ' + scatSet['lutPath']
                lutFiles = glob(scatSet['lutPath']+'DDA_LUT_plate_freq*.nc') 
                
                listFreq = [l.split('DDA_LUT_plate_')[1].split('_elv')[0].split('freq')[1] for l in lutFiles]
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('elv')[1].split('.nc')[0] for l in lutFiles]
                listElev = list(dict.fromkeys(listElev))
                dicSettings['scatSet']['lutFreqMono'] = [float(f) for f in listFreq]
                dicSettings['scatSet']['lutElevMono'] = [int(e) for e in listElev]
                dicSettings['scatSet']['LUTFilesMono'] = lutFiles
                #- now same for aggregates
                lutFiles = glob(scatSet['lutPath']+'DDA_LUT_dendrite_aggregates_freq*.nc') 
                #listFreq = [l.split('LUT_dendrite_aggregates')[1].split('_elv')[0].split('freq')[1] for l in lutFiles]
                listFreq = [l.split('DDA_LUT_dendrite_aggregates_freq')[1].split('_elv')[0] for l in lutFiles] 
                listFreq = list(dict.fromkeys(listFreq))
                listElev = [l.split('elv')[1].split('.nc')[0] for l in lutFiles] #TODO after testing remove _log
                listElev = list(dict.fromkeys(listElev))
                dicSettings['scatSet']['lutFreqAgg'] = [float(f) for f in listFreq]
                dicSettings['scatSet']['lutElevAgg'] = [int(e) for e in listElev]
                dicSettings['scatSet']['LUTFilesAgg'] = lutFiles
                
                
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
                 


