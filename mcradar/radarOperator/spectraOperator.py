# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import mcradar as mcr
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


def getMultFrecSpec(wls, elvs, mcTable, velBins, velCenterBins , centerHeight, convolute,nave,noise_pow,eps_diss,uwind,time_int,theta,scatSet={'mode':'full', 'safeTmatrix':False}):
    """
    Calculation of the multi-frequency spectrograms 
    
    Parameters
    ----------
    wls: wavelenght (iterable) [mm]
    elvs: elevation (iterable) [°]
    mcTable: McSnow output returned from calcParticleZe()
    velBins: velocity bins for the calculation of the spectrogram (array) [m/s]
    velCenterBins: center of the velocity bins (array) [m/s]
    centerHeight: center height of each range gate (array) [m]
    
    Returns
    -------
    xarray dataset with the multi-frequency spectrograms
    xarray dims = (range, vel) 
    """
    
    specTable = xr.Dataset()
    if (scatSet['mode'] == 'SSRGA') or (scatSet['mode'] == 'Rayleigh') or (scatSet['mode'] == 'SSRGA-Rayleigh'):
        mcTable['sZeMultH'] = mcTable['sZeH'] * mcTable['sMult']
        
        intSpec = mcTable.groupby_bins("vel", velBins).sum()
        specTable['spec_H'] = group['sZeMultH'].rename({'vel_bins':'vel'}).assign_coords({'vel':velCenterBins})
        if convolute == True:
            for wl in wls:
                for elv in elvs:
                    specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv),wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
    
    else:
        mcTable['sZeMultH'] = mcTable['sZeH'] * mcTable['sMult']
        mcTable['sZeMultV'] = mcTable['sZeV'] * mcTable['sMult']
        group = mcTable.groupby_bins("vel", velBins,labels=velCenterBins).sum()#.sel(wavelength=wl,elevation=elv).groupby_bins("vel", velBins,labels=velCenterBins).sum()#.rename({'vel_bins':'doppler_vel'})
            
        specTable['spec_H'] = group['sZeMultH'].rename({'vel_bins':'vel'})#.assign_coords({'vel':velCenterBins})
        specTable['spec_V'] = group['sZeMultV'].rename({'vel_bins':'vel'})
        if convolute == True:
            for wl in wls:
                for elv in elvs:
                    mcTablePD = mcTable.sel(wavelength=wl,elevation=elv)
                    #.assign_coords({'vel':velCenterBins})
                    #print(specTable)
                    #quit()
                    specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
                    specTable['spec_V'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
                    #spec_H = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
                    #fig,ax = plt.subplots(ncols=3,figsize=(15,5))
                    #ax[0].plot(mcTable.vel,10*np.log10(mcTable.sZeMultH.sel(wavelength=mcTable.wavelength[0],elevation=30)),marker='.',ls='None')
                    #ax[1].plot(mcTable.vel,10*np.log10(mcTable.sZeH.sel(wavelength=mcTable.wavelength[0],elevation=30)),marker='.',ls='None')
                    #ax[2].plot(specTable.vel,10*np.log10(spec_H),marker='.',ls='None')
                    #ax[0].set_ylabel('sZe Mult [dB]')
                    #ax[0].set_xlabel('vel')
                    #ax[1].set_ylabel('sZe single [dB]')
                    #ax[1].set_xlabel('vel')
                    #ax[2].set_ylabel('sZe [dB]')
                    #ax[2].set_xlabel('vel')
                    #plt.tight_layout()
                    #plt.savefig('new_McRadar_lowestheight_noloop.png')
                    #plt.show()
                    #quit()
                    #specTable['spec_V'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V'].sel(wavelength=wl,elevation=elv),wl,velCenterBins,eps_diss,noise_pow,nave,theta,uwind,time_int,centerHeight)
                    #mcr.lin2db(specH).plot()
                    #plt.show()
                    #quit()
    specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
    
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
    
    return spectrum#pd.DataFrame(data=spectrum,index=vel)
    
    
