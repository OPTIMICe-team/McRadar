# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import mcradar as mcr
import scipy.signal as sig
from scipy import constants
#from numba import jit
import timeit
#@jit(nopython=True)
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



def getMultFrecSpec(wls, elvs, mcTable, velBins, velCenterBins , centerHeight, 
					convolute, nave, noise_pow, eps_diss, uwind, time_int, theta,
					k_theta, k_phi, k_r, tau, scatSet={'mode':'full', 'safeTmatrix':False}):

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
    convolute: if True, noise and spectral broadening will be convoluted to the spectrum
    nave: number of spectral averages
    noise_pow: noise power of radar 
    eps_diss: eddy dissipation rate m/s²
    uwind: np.array containing x,y,z component of wind velocity
    time_int: integration time of radar in sec 
    theta: beamwidth of radar in rad
    k_theta: wind shear in theta direction (when looking zenith this is in x direction)
    k_phi: wind shear in phi direction (when looking zenith this is in y direction)
    k_r: wind shear in r direction (when looking zenith this is in z direction)
    tau: pulse width
    
    Returns
    -------
    xarray dataset with the multi-frequency spectrograms
    xarray dims = (range, vel) 
    """
    
    divAggMono = True # TODO make that a keyword!
    specTable = xr.Dataset()
    if (scatSet['mode'] == 'SSRGA') or (scatSet['mode'] == 'Rayleigh') or (scatSet['mode'] == 'SSRGA-Rayleigh'):
        mcTable['sZeMultH'] = mcTable['sZeH'] * mcTable['sMult']
        
        group = mcTable.groupby_bins('vel', velBins,labels=velCenterBins).sum()
        specTable['spec_H'] = group['sZeMultH'].rename({'vel_bins':'vel'})#.assign_coords({'vel':velCenterBins})
        #specTable['specBroad_H'] = xr.Dataarray(dims=['elevation','wavelength'],coords={'elevation':specTable.elevation,'wavelength':specTable.wavelength})
        
        if convolute == True:
            for wl,th,nv,noise in zip(wls,theta,nave,noise_pow):
                for elv in elvs:
                    specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                       noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
    	
    else:
        mcTable['sZeMultH'] = mcTable['sZeH'] * mcTable['sMult']
        mcTable['sZeMultV'] = mcTable['sZeV'] * mcTable['sMult']
        mcTable['sZeMultHV'] = mcTable['sZeHV'] * mcTable['sMult']
        group = mcTable.groupby_bins('vel', velBins,labels=velCenterBins).sum()#.sel(wavelength=wl,elevation=elv).groupby_bins("vel", velBins,labels=velCenterBins).sum()#.rename({'vel_bins':'doppler_vel'})
        
        specTable['spec_H'] = group['sZeMultH'].rename({'vel_bins':'vel'})#.assign_coords({'vel':velCenterBins})
        specTable['spec_V'] = group['sZeMultV'].rename({'vel_bins':'vel'})
        specTable['spec_HV'] = group['sZeMultHV'].rename({'vel_bins':'vel'})
        #specTable['specBroad_H'] = xr.Dataarray(dims=['elevation','wavelength'],coords={'elevation':specTable.elevation,'wavelength':specTable.wavelength})
        #specTable['specBroad_V'] = xr.Dataarray(dims=['elevation','wavelength'],coords={'elevation':specTable.elevation,'wavelength':specTable.wavelength})
        #specTable['specBroad_HV'] = xr.Dataarray(dims=['elevation','wavelength'],coords={'elevation':specTable.elevation,'wavelength':specTable.wavelength})

		
        if convolute == True:
            for wl,th,nv,noise in zip(wls,theta,nave,noise_pow):
            	
                for elv in elvs:
                    if elv == 30:
                        nv = nave[3]
                    #mcTablePD = mcTable.sel(wavelength=wl,elevation=elv)
                    #specTable['spec_H'].loc[:,elv,wl],specTable['specBroad_H'].loc[elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                    #                                                  noise_pow,nave,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                    specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                      noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                    specTable['spec_V'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                      noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                    specTable['spec_HV'].loc[:,elv,wl] = convoluteSpec(specTable['spec_HV'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                       noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)

        if divAggMono:
            mcTableMono = mcTable.where(mcTable['sNmono']==1,drop=True) # select only plates
            mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True)
            if len(mcTableMono.sPhi)>0:
                mcTableMono['sZeMultH'] = mcTableMono['sZeH'] * mcTableMono['sMult']
                mcTableMono['sZeMultV'] = mcTableMono['sZeV'] * mcTableMono['sMult']
                mcTableMono['sZeMultHV'] = mcTableMono['sZeHV'] * mcTableMono['sMult']
                groupMono = mcTableMono.groupby_bins('vel', velBins,labels=velCenterBins).sum()
                specTable['spec_H_Mono'] = groupMono['sZeMultH'].rename({'vel_bins':'vel'})#.assign_coords({'vel':velCenterBins})
                specTable['spec_V_Mono'] = groupMono['sZeMultV'].rename({'vel_bins':'vel'})
                specTable['spec_HV_Mono'] = groupMono['sZeMultHV'].rename({'vel_bins':'vel'})
                if convolute == True:
                    for wl,th,nv,noise in zip(wls,theta,nave,noise_pow):
                        for elv in elvs:
                            if elv == 30:
                                nv = nave[3]
                            #specTable['spec_H'].loc[:,elv,wl],specTable['specBroad_H'].loc[elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                            #                                                  noise_pow,nave,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                            specTable['spec_H_Mono'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H_Mono'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                          noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                            specTable['spec_V_Mono'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V_Mono'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                          noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                            specTable['spec_HV_Mono'].loc[:,elv,wl] = convoluteSpec(specTable['spec_HV_Mono'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                           noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)

            if len(mcTableAgg.sPhi)>0:
                mcTableAgg['sZeMultH'] = mcTableAgg['sZeH'] * mcTableAgg['sMult']
                mcTableAgg['sZeMultV'] = mcTableAgg['sZeV'] * mcTableAgg['sMult']
                mcTableAgg['sZeMultHV'] = mcTableAgg['sZeHV'] * mcTableAgg['sMult']
                groupAgg = mcTableAgg.groupby_bins('vel', velBins,labels=velCenterBins).sum()
                specTable['spec_H_Agg'] = groupAgg['sZeMultH'].rename({'vel_bins':'vel'})#.assign_coords({'vel':velCenterBins})
                specTable['spec_V_Agg'] = groupAgg['sZeMultV'].rename({'vel_bins':'vel'})
                specTable['spec_HV_Agg'] = groupAgg['sZeMultHV'].rename({'vel_bins':'vel'})

                if convolute == True:
                    for wl,th,nv,noise in zip(wls,theta,nave,noise_pow):
                        for elv in elvs:
                            if elv == 30:
                                nv = nave[3]
                            specTable['spec_H_Agg'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H_Agg'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                          noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                            specTable['spec_V_Agg'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V_Agg'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                          noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)
                            specTable['spec_HV_Agg'].loc[:,elv,wl] = convoluteSpec(specTable['spec_HV_Agg'].sel(wavelength=wl,elevation=elv).fillna(0),wl,velCenterBins,eps_diss,
                                                                           noise,nv,th,uwind,time_int,centerHeight,k_theta,k_phi,k_r,tau)

    specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
    return specTable

def convoluteBroadfft(spec,vel,specBroad):
    """
     use fft to do convolution of broadening terms
     
     Parameters:
     -----------
     spec: spectrum without broadening
     vel: Doppler vel array
     specBroad: broadening to be added
     
     Returns:
     --------
     spectrum with added turbulence
    """
    prefactor_turb = 1.0 / (np.sqrt(2.0 * np.pi) * specBroad)
    turb = np.zeros(len(vel))
    dv = np.diff(vel)[0]

    for i in range(len(vel)):
        #gaussian function with same length as radar spectrum, centered around zero
        turb[i] = np.exp(-1*(vel[i]-0)**2.0/(2.0*specBroad**2.0))*dv

    spec_turb = sig.fftconvolve(spec.values,turb,mode='same')
    return spec_turb * prefactor_turb
    

def convoluteNoise(spec,vel,noise_pow,nave):
    """
     convolution of noise
     
     Parameters:
     -----------
     spec: spectrum without noise 
     vel: Doppler vel array
     noise_pow: noise power [mm^6/m^3] 
     nave: number of spectral averages
     
     Returns:
     --------
     spectrum with added noise
    """	
    dv = np.diff(vel)[0]
    Ni = noise_pow / (len(vel) * dv)
    random_numbers = np.random.uniform(size=len(vel)*nave)
    S_bin_noise = np.zeros(len(vel))
    for iave in range(nave):
        S_bin_noise = S_bin_noise + (-np.log(random_numbers[iave * (len(vel)) : ((iave+1) * len(vel))]) * (spec + np.ones(len(vel))*Ni )) 

    return S_bin_noise / nave
    
def convoluteTurb(spec,spec_turb,vel,specBroad,dv):
#- turbulence convolution:
    """
    original convolution, without fft. This is several orders of magnitudes longer than convoluteBroadfft!!!
    """
    prefactor_turb = 1.0 / (np.sqrt(2.0 * np.pi) * specBroad)
    for i in range(len(vel)):
        integral = 0
        for ii in range(len(vel)):
            exp_arg = (-1.0*(vel[i]-vel[ii])**2)/(2.0*specBroad**2)
            if exp_arg >= -100:
                integral = integral + (spec.values[ii]*np.exp(exp_arg)*dv)
        spec_turb[i] = prefactor_turb * integral    
    return spec_turb

# TODO: only generate one wind shear height, otherwise no windshear!! Or other solution, however, right now wind shear gets added to entire profile... Or: make wind shear, uwind and eps height dependent
def convoluteSpec(spec,wl,vel,eps,noise_pow,nave,theta,u_wind,time_int,height,k_theta,k_phi,k_r,tau,PSD=False):
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
    theta: beamwidth of radar in rad
    u_wind: horizontal wind velocity in m/s
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
    
    
    L_s = u_wind*time_int + 2*height*np.sin(theta)
    L_lam = (wl*1e-3)/2    
    sigma_t2 = 3/4*(eps/(2*np.pi))**(2/3)*( L_s**(2/3) - L_lam**(2/3) ) # turbulence broadening term (3*kolmogorov/2, where kolmogorov=0.5, therefore 3/4)
    
    #finite beamwidth broadening
    sigma_b2 = u_wind**2*theta**2/2.76 
    # windshear broadening according to Doviak and Zrnic 1993
    sigma_theta = theta/(4*np.sqrt(np.log(2)))
    sigma_stheta = height*sigma_theta*k_theta # x-component (in case of 90° elevation)
    sigma_sphi = height*sigma_theta*k_phi # y-component (in case of 90° elevation)
    sigma_sr = 0.35*constants.c*tau/2*k_r # z-component (in case of 90° elevation)
    sigma_s2 = sigma_stheta**2 + sigma_sphi**2 + sigma_sr**2
    
    # all broadening terms  
    specBroad = np.sqrt(sigma_t2 + sigma_b2 + sigma_s2)
    # convolute spectrum with turbulence
    spec_turb = convoluteBroadfft(spec,vel,specBroad)
    
    #- convolute random noise
    spectrum = convoluteNoise(spec_turb,vel,noise_pow,nave)
    
    if PSD == True:
    	return spectrum, specBroad
    else:
    	return spectrum
    
