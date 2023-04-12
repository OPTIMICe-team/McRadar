# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import mcradar as mcr
import scipy.signal as sig
from scipy import constants
import timeit
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



def getMultFrecSpec(wls, elvs, mcTable, velBins, velCenterBins , centerHeight, convolute,nave,noise_pow,eps_diss,uwind,time_int,theta,variable_theta,scatSet={'mode':'full', 'safeTmatrix':False}):

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
                    specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv),wl,velCenterBins,eps_diss,
                                                                      noise_pow,nave,theta,uwind,time_int,centerHeight,variable_theta)
    
    else:
        mcTable['sZeMultH'] = mcTable['sZeH'] * mcTable['sMult']
        mcTable['sZeMultV'] = mcTable['sZeV'] * mcTable['sMult']
        mcTable['sZeMultHV'] = mcTable['sZeHV'] * mcTable['sMult']
        group = mcTable.groupby_bins('vel', velBins,labels=velCenterBins).sum()#.sel(wavelength=wl,elevation=elv).groupby_bins("vel", velBins,labels=velCenterBins).sum()#.rename({'vel_bins':'doppler_vel'})
        
        specTable['spec_H'] = group['sZeMultH'].rename({'vel_bins':'vel'})#.assign_coords({'vel':velCenterBins})
        specTable['spec_V'] = group['sZeMultV'].rename({'vel_bins':'vel'})
        specTable['spec_HV'] = group['sZeMultHV'].rename({'vel_bins':'vel'})
        if convolute == True:
            for wl in wls:
                for elv in elvs:
                    mcTablePD = mcTable.sel(wavelength=wl,elevation=elv)
                    specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                      noise_pow,nave,theta,uwind,time_int,centerHeight,variable_theta)
                    specTable['spec_V'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                      noise_pow,nave,theta,uwind,time_int,centerHeight,variable_theta)
                    specTable['spec_HV'].loc[:,elv,wl] = convoluteSpec(specTable['spec_HV'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                       noise_pow,nave,theta,uwind,time_int,centerHeight,variable_theta)

    specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
    return specTable

def convoluteTurbfft(spec,spec_turb,turb,vel,specBroad,dv):
    prefactor_turb = 1.0 / (np.sqrt(2.0 * np.pi) * specBroad)
    
    for i in range(len(vel)):
        #gaussian function with same length as radar spectrum, centered around zero
        turb[i] = np.exp(-1*(vel[i]-0)**2.0/(2.0*specBroad**2.0))*dv
    
    spec_turb = sig.fftconvolve(spec.values,turb,mode='same')
    return spec_turb * prefactor_turb
    #return spec_turb    
def convoluteTurb(spec,spec_turb,vel,specBroad,dv):
#- turbulence convolution:
    prefactor_turb = 1.0 / (np.sqrt(2.0 * np.pi) * specBroad)
    for i in range(len(vel)):
        integral = 0
        for ii in range(len(vel)):
            exp_arg = (-1.0*(vel[i]-vel[ii])**2)/(2.0*specBroad**2)
            if exp_arg >= -100:
                integral = integral + (spec.values[ii]*np.exp(exp_arg)*dv)
        spec_turb[i] = prefactor_turb * integral    
    return spec_turb

# TODO: wind needs to have 3D components for finite beam width, wind shear and turbulence. For the 30° elevation this is necessary
def convoluteSpec(spec,wl,vel,eps,noise_pow,nave,theta,u_wind,time_int,height,variable_theta):
    """
    this function convolutes the spectrum with turbulence and adds random noise, optional!
    
    Parameters
    ----------
    spec: spectral data (xarray.dataarray) [mm^6/m^3]
    wl: wavelength in mm
    vel: Doppler velocity array (m/s)
    eps: eddy dissipation rate m/s²
    noise_pow: radar noise power [mm^6/m^3] 
    nave: number of spectral averages
    theta: beamwidth of radar
    u_wind: vertical wind velocity in m/s
    time_int: integration time of radar in sec 
    height: centre height of range gate    
    k_theta: wind shear in theta direction (when looking zenith this is in x direction)
    k_phi: wind shear in phi direction (when looking zenith this is in y direction)
    k_r: wind shear in r direction (when looking zenith this is in z direction)
    tau: pulse width
    
    Returns
    -------
    spectrum with added noise and turbulence broadening as np.array 
    """
    if variable_theta:
        # since our X-Band has a different beamwidth, I am doing this!
        f = constants.c/wl*1e3
        if f == 9.6e9:
            theta = 1.0/2./180.*np.pi
    
    L_s = u_wind*time_int + 2*height*np.sin(theta)
    L_lam = (wl*1e-3)/2    
    sigma_t2 = 3/4*(eps/(2*np.pi))**(2/3)*( L_s**(2/3) - L_lam**(2/3) ) # turbulence broadening term (3*kolmogorov/2, where kolmogorov=0.2, therefore 3/4)
    
    #finite beamwidth broadening
    sigma_b2 = u_wind**2*theta**2/2.76 # TODO check difference in pamtra and Alessandro presentation, TODO: is u_wind vertical or horizontal wind?
    # windshear broadening
    #sigma_theta = theta/(4*np.sqrt(np.log(2)))
    #sigma_stheta = height*sigma_theta*k_theta
    #sigma_sphi = r6*sigma_theta*k_phi
    #sigma_sr = 0.35*c*tau/2*k_r
    #sigma_s2 = sigma_stheta**2 + sigma_sphi**2 + sigma_sr**2

    # all broadening terms    
    specBroad = np.sqrt(sigma_t2 + sigma_b2)# + sigma_s2)
    
    spec_turb = np.zeros(len(vel))#spec.copy()*np.NaN
    turb = np.zeros(len(vel))
    dv = np.diff(vel)[0]
    # convolute spectrum with turbulence    
    spec_turb = convoluteTurbfft(spec,spec_turb,turb,vel,specBroad,dv)
    
    #print('time fft: ',timeit.timeit(lambda: convoluteTurbfft(spec,spec_turb,turb,vel,specBroad,dv),number=10))
    #print('time loop: ',timeit.timeit(lambda: convoluteTurb(spec,spec_turb,vel,specBroad,dv),number=10))
    #quit()
    
    #- convolute random noise
    Ni = noise_pow / (len(vel) * dv)
    random_numbers = np.random.uniform(size=len(vel)*nave)
    S_bin_noise = np.zeros(len(vel))
    for iave in range(nave):
        S_bin_noise = S_bin_noise + (-np.log(random_numbers[iave * (len(vel)) : ((iave+1) * len(vel))]) * (spec_turb + np.ones(len(vel))*Ni )) 

    spectrum = S_bin_noise / nave
    
    return spectrum#pd.DataFrame(data=spectrum,index=vel)
    
    
