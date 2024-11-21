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
                 convolute=True,nave=np.array([10,20,28,90]),noise_pow=np.array([-50,-63,-58]),
                 theta=np.array([1.0,0.6,0.6]) , time_int=2.0 , tau=143*1e-9 ,
                 uwind=10.0, eps_diss=np.array([1e-6]), k_theta=np.array([0]),k_phi=np.array([0]),k_r=np.array([0]),shear_height0=0,shear_height1=0,
                 maxVel=3, minVel=-3,velVec=None,
                 freq=np.array([9.5e9, 35e9, 95e9]),
                 maxHeight=5500, minHeight=0,
                 heightRes=50, gridBaseArea=1,
                 attenuation=False,onlyIce=True,
                 beta=0,beta_std=0,
                 scatSet={'mode':'DDA', 'selmode':'KNeighborsRegressor', 'n_neighbors':5, 'radius':1e-10,
                          'safeTmatrix':False,'K2':0.93,'orientational_avg':False}):
    
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
    nave: number of spectral averages (default = 10, 20, 28, 90 for X-Band, Ka-Band, W-Band and pol. W-Band), needed only if convolute == True
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
    freq: radar frequency (default = 9.5e9, 35e9, 95e9) [Hz]
    maxHeight: maximum height (default = 5500) [m]
    minHeight: minimun height (default = 0) [m]
    heightRes: resolution of the height bins (default = 50) [m]
    gridBaseArea: area of the grid base (default = 1) [m^2]
    attenuation: get path integrated attenuation based on water vapour, O2 and H2O. This uses PAMTRA
    onlyIce: if the atmo file goes to warmer temperatures, only go until 0°C and remove everything that is warmer!
    beta: wobbling angle of particles, default: 0°
    beta_std: standard deviation of the wobbling angle of particles, default: 0°. For the wobbling of particles, a normal distribution with mean beta and std beta_std is used.
    scatSet: dictionary that defines the settings for the scattering calculations
    scatSet['mode']: string that defines the scattering mode. Valid values are
                        - Tmatrix -> pytmatrix calculations for each superparticle
                        - DDA -> this mode uses DDA table. Possible frequencies: 9.6GHz, 35.5GHz, 94.0GHz. Selection is based on mass, ar and size. 
                        		 Columnar or plate-like scattering table will be chosen depending on the aspect ratio of the particles. You need to specify the path to the LUT.  
    scatSet['K2']: dielectric constant of the particles, default: 0.93
    scatSet['selmode']: string that defines the selection mode for the DDA database. Valid entries are 
                        - KNeighborsRegressor: this mode uses the KNeighborsRegressor from sklearn. The n_neighbors closest neighbors in Dmax, aspect ratio and mass are selected 
                                                and the corresponding scattering properties are averaged based on the inverse distance of the neighbors.  
                        - NearestNeighbors: the n_neighbors closest neighbors in Dmax, aspect ratio and mass are selected and the scattering properties of these points are averaged.
                        - radius: this mode uses the radius_neighbors from sklearn. All neighbors (Dmax, aspect ratio, mass) within the predefined radius are selected and the scattering properties are averaged.
    scatSet['n_neighbors']: number of neighbors to use for the KNeighborsRegressor
    scatSet['radius']: radius in which the nearest neighbours are selected when scatSet['selmode'] is set to radius.
    scatSet['lutPath']: path to where the DDA calculations are stored. This is only needed when scatSet['mode'] is set to DDA.
    scatSet['ndgs']: number of division points used to integrate over the particle surface (default = 30) when using Tmatrix as scattering mode
    Returns
    -------
    dicSettings: dictionary with all parameters
    for starting the caculations
    """
    if 'mode' not in scatSet.keys():
      scatSet['mode'] = 'DDA'
    if 'safeTmatrix' not in scatSet.keys():
      scatSet['safeTmatrix'] = False
    if 'K2' not in scatSet.keys():
        scatSet['K2'] = 0.93
    if 'ndgs' not in scatSet.keys():
        scatSet['ndgs'] = 30
    if 'n_neighbors' not in scatSet.keys():
        scatSet['n_neighbors'] = 5
    if 'radius' not in scatSet.keys():
        scatSet['radius'] = 1e-10
    if 'selmode' not in scatSet.keys():
        scatSet['selmode'] = 'KNeighborsRegressor'
    if 'orientational_avg' not in scatSet.keys():
        scatSet['orientational_avg'] = False

    if dataPath != None:
        
        del_v = (maxVel-minVel) / nfft
        dicSettings = {'dataPath':dataPath,
                       'elv':elv,
                       #'nfft':nfft,
                       #'maxVel':maxVel,
                       #'minVel':minVel,
                       #'velRes':(maxVel - minVel)/nfft,
                       'freq':freq,
                       'wl':(constants.c / freq) * 1e3, #[mm]
                       'maxHeight':maxHeight,
                       'minHeight':minHeight,
                       'heightRes':heightRes,
                       'heightRange':np.arange(minHeight, maxHeight, heightRes),
                       'gridBaseArea':gridBaseArea,
                       'scatSet':scatSet,
                       'convolute':convolute,
                       'attenuation':attenuation,
                       'nave':nave,
                       #'noise_pow':(10**(noise_pow/10))*(nfft*del_v),
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
                       'onlyIce':onlyIce,
                       'beta':beta,
                        'beta_std':beta_std,
                       }

        if velVec.any():
            dicSettings['velBins'] = velVec
            dicSettings['velCenterBin'] = velVec[0:-1]+np.diff(velVec)/2.
            dicSettings['velRes'] = np.diff(velVec)[0]
            dicSettings['nfft'] = len(velVec)-1
            dicSettings['noise_pow']=(10**(noise_pow/10))*(dicSettings['nfft']*dicSettings['velRes'])

        else:

            velBins = np.arange(minVel, maxVel, dicSettings['velRes'])
            velCenterBin = velBins[0:-1]+np.diff(velBins)/2.

            dicSettings['velBins']=velBins
            dicSettings['velCenterBin']=velCenterBin
            dicSettings['nfft']=nfft
            dicSettings['maxVel']=maxVel
            dicSettings['minVel']=minVel
            dicSettings['velRes']=(maxVel - minVel)/nfft
            dicSettings['noise_pow']=(10**(noise_pow/10))*(dicSettings['nfft']*dicSettings['velRes'])

        if onlyIce == True:
            print(atmoFile)
            if atmoFile != None: 
                atmo = np.loadtxt(atmoFile)
                height = atmo[:,0]
                temp = atmo[:,2]# -273.15
                atmoPD = pd.DataFrame(data=temp,index=height,columns=['temp'])
                atmoPD.index.name='range'
                atmoPD['press'] = atmo[:,3]
                atmoPD['relHum'] = atmo[:,6]
                atmoXR = atmoPD.to_xarray()
                atmoReindex = atmoXR.reindex({'range':dicSettings['heightRange']+dicSettings['heightRes']/2},method='nearest')
                dicSettings['temp']=atmoReindex.temp
                dicSettings['relHum']=atmoReindex.relHum
                dicSettings['press']=atmoReindex.press
            else:
                raise FileNotFoundError('since you want to check for melted particles (onlyIce==True) you need to give an atmoFile as input.')
    elif PSD == True:
        del_v = (maxVel-minVel) / nfft
		
        dicSettings = {'elv':elv,
                       'nfft':nfft,
                       'maxVel':maxVel,
                       'minVel':minVel,
                       'velRes':(maxVel - minVel)/nfft,
                       'freq':freq,
                       'wl':(constants.c / freq) * 1e3, #[mm]
                       'maxHeight':maxHeight,
                       'minHeight':minHeight,
                       'heightRes':heightRes,
                       'heightRange':np.arange(minHeight, maxHeight, heightRes),
                       'gridBaseArea':gridBaseArea,
                       'scatSet':scatSet,
                       'convolute':convolute,
                       'nave':nave,
                       'noise_pow':(10**(noise_pow/10))*(nfft*del_v),#noise_pow,
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
                       'onlyIce':onlyIce,
                       'beta':beta,
                        'beta_std':beta_std,
                       }

        velBins = np.arange(minVel, maxVel, dicSettings['velRes'])
        velCenterBin = velBins[0:-1]+np.diff(velBins)/2.
		#radar_Pnoise = 
        dicSettings['velBins']=velBins
        dicSettings['velCenterBin']=velCenterBin
    else:
        raise ValueError('No valid data are provided. Either give the path to the McSnow output (use the dataPath parameter e.g. loadSettings(dataPath="/data/path/.") or set PSD=True')
        
        
    #print(attenuation)
    if attenuation == True:
        if not (('temp' in dicSettings) and ('relHum' in dicSettings) and ('press' in dicSettings)):
            if atmoFile != None: 
                atmo = np.loadtxt(atmoFile)
                height = atmo[:,0]
                temp = atmo[:,2]# -273.15
                atmoPD = pd.DataFrame(data=temp,index=height,columns=['temp'])
                atmoPD.index.name='range'
                atmoPD['press'] = atmo[:,3]
                atmoPD['relHum'] = atmo[:,6]
                atmoXR = atmoPD.to_xarray()
                atmoReindex = atmoXR.reindex({'range':dicSettings['heightRange']+dicSettings['heightRes']/2},method='nearest')
                dicSettings['temp']=atmoReindex.temp
                dicSettings['relHum']=atmoReindex.relHum
                dicSettings['press']=atmoReindex.press
                
            else:
                raise FileNotFoundError(['since you want to do the attenuation correction you need to give an atmoFile as input.'])
    
    if scatSet['mode'] == 'DDA':
        print(scatSet) 
        print(scatSet['selmode'])
        print('you selected DDA as scattering mode. The scattering properties are selected from a LUT by choosing the closest neighbors in Dmax, aspect ratio and mass using the method {}'.format(scatSet['selmode']))
        
        if 'lutPath' not in scatSet.keys():
            raise FileNotFoundError('with this scattering mode ', scatSet['mode'], 'a valid path to the scattering LUT is required.')
            
        elif not os.path.exists(scatSet['lutPath']):    
            raise FileNotFoundError(scatSet['lutPath'], 'is not valid, check your settings')
            
        
        #if float(dicSettings['freq']) != float(94e9):
        #    msg = ('\n').join(['with this scattering mode ', scatSet['mode'],
        #                       'only freq=94.00GHz is possible!',
        #                       'check your settings'])
        #    dicSettings = None
        
        
    

    return dicSettings
                 


