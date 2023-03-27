# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getVelIntSpec(mcTable, mcTable_binned, variable):
    """
    Calculates the integrated reflectivity for each velocity bin
    
    Parameters
    ----------
    mcTable: McSnow output returned from calcParticleZe()
    mcTable_binned: McSnow table output binned for a given velocity bin
    variable: name of column variable wich will be integrated over a velocity bin
    
    Returns
    -------
    mcTableVelIntegrated: table with the integrated reflectivity for each velocity bin
    """

    mcTableVelIntegrated = mcTable.groupby(mcTable_binned)[variable].agg(['sum'])
    
    return mcTableVelIntegrated


def getMultFrecSpec(wls,elvs, mcTable, velBins, velCenterBins , centerHeight, convolute,nave,noise_pow,eps_diss,uwind,time_int,theta,scatSet={'mode':'full', 'safeTmatrix':False}):
    """
    Calculation of the multi-frequency spectrograms 
    
    Parameters
    ----------
    wls: wavelenght (iterable) [mm]
    mcTable: McSnow output returned from calcParticleZe()
    velBins: velocity bins for the calculation of the spectrogram (array) [m/s]
    velCenterBins: center of the velocity bins (array) [m/s]
    centerHeight: center height of each range gate (array) [m]
    
    Returns
    -------
    xarray dataset with the multi-frequency spectrograms
    xarray dims = (range, vel) 
    """
    
    mcTable_binned = pd.cut(mcTable['vel'], velBins)

    tmpDataDic = {}
     
    for wl in wls:
        for elv in elvs:
            wlStr = '{:.2e}'.format(wl)
            
            if (scatSet['mode'] == 'SSRGA') or (scatSet['mode'] == 'Rayleigh') or (scatSet['mode'] == 'SSRGA-Rayleigh'):
              mcTable['sZeMultH_{0}_elv{1}'.format(wlStr,elv)] = mcTable['sZeH_{0}_elv{1}'.format(wlStr,elv)] * mcTable['sMult']
              #print(mcTable['sMult'])
              #plt.plot(mcTable['sZeMultH_{0}'.format(wlStr)],mcTable['radii_mm'],label='Mult')
              #plt.plot(mcTable['sZeH_{0}'.format(wlStr)],mcTable['radii_mm'],label='sZe')
              #plt.legend()
              #plt.show()
              intSpecH = getVelIntSpec(mcTable, mcTable_binned,'sZeMultH_{0}_elv{1}'.format(wlStr,elv))
              if convolute == True:
                  intSpecH = convoluteSpec(intSpecH,wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
              tmpDataDic['spec_H_{0}'.format(wlStr)] = intSpecH.values[:,0]
            
            else:
              mcTable['sZeMultH_{0}_elv{1}'.format(wlStr,elv)] = mcTable['sZeH_{0}_elv{1}'.format(wlStr,elv)] * mcTable['sMult']
              mcTable['sZeMultV_{0}_elv{1}'.format(wlStr,elv)] = mcTable['sZeV_{0}_elv{1}'.format(wlStr,elv)] * mcTable['sMult']

              intSpecH = getVelIntSpec(mcTable, mcTable_binned,'sZeMultH_{0}_elv{1}'.format(wlStr,elv))
              intSpecV = getVelIntSpec(mcTable, mcTable_binned, 'sZeMultV_{0}_elv{1}'.format(wlStr,elv))
              if convolute == True:
                  intSpecH = convoluteSpec(intSpecH,wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
                  intSpecV = convoluteSpec(intSpecV,wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
              tmpDataDic['spec_H_{0}_elv{1}'.format(wlStr,elv)] = intSpecH.values[:,0]
              tmpDataDic['spec_V_{0}_elv{1}'.format(wlStr,elv)] = intSpecV.values[:,0]
        
        

    #converting to XR
    specTable = pd.DataFrame(data=tmpDataDic, index=velCenterBins)
    specTable = specTable.to_xarray()
    specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
    specTable = specTable.rename_dims(dims_dict={'index':'vel'}).rename(name_dict={'index':'vel'})
    
    return specTable
    
def convoluteSpec(spec,wl,vel,eps,noise_pow,nave,theta,u_wind,time_avg,height):
    """
    this function convolutes the spectrum with turbulence and adds random noise, optional!
    Parameters
    ----------
    spec: spectral data (pd.dataframe) [mm^6/m^3]
    sigma_t: spectrum width due to turbulence
    np: radar noise power [mm^6/m^3]
    
    Returns
    -------
    convoluted and noisy spectrum as pd.dataframe with the index of the input spec 
    """

    L_s = u_wind*time_avg + 2*height*np.sin(theta)
    L_lam = wl/2    
    sigma_t2 = 3/4*(eps/2*np.pi)**(2/3)*( L_s**(2/3) - L_lam**(2/3) )
    
    spec_turb = np.zeros(len(vel))#spec.copy()*np.NaN
    dv = np.diff(vel)[0]
    prefactor_turb = 1.0 / (np.sqrt(2.0 * np.pi) * np.sqrt(sigma_t2))
    #- turbulence convolution:
    for i in range(len(vel)):
        integral = 0
        for ii in range(len(vel)):
            exp_arg = (-1.0*(vel[i]-vel[ii])**2)/(2.0*sigma_t2)
            if exp_arg >= -100:
                integral = integral + (spec.values[ii]*np.exp(exp_arg)*dv)
        spec_turb[i] = prefactor_turb * integral
    
    #- add random noise
    Ni = noise_pow / (len(vel) * dv)
    
    random_numbers = np.random.uniform(size=len(vel)*nave)
    
    S_bin_noise = np.zeros(len(vel))
    
    for iave in range(nave):
        S_bin_noise = S_bin_noise + (-np.log(random_numbers[iave * (len(vel)) : ((iave+1) * len(vel))]) * (spec_turb + np.ones(len(vel))*Ni )) 
    spectrum = S_bin_noise / nave
    return pd.DataFrame(data=spectrum,index=spec.index)
    
    
