# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import pandas as pd
import numpy as np
import xarray as xr

## it can be more general allowing the user to pass
## the name of the columns
def getMcSnowTable(mcSnowPath):
    """
    Read McSnow output table
    
    Parameters
    ----------
    mcSnowPath: path for the output from McSnow
    
    Returns
    -------
    Pandas DataFrame with the McSnow output variables. This DataFrame additionally includes
    a column for the radii and the density [sRho]. The 
    velocity is negative towards the ground. 

    """
    
    #open nc file with xarray
    mcTableXR = xr.open_dataset(mcSnowPath)
    print(mcTableXR)
    #change to pandas dataframe, since McRadar has been working with that
    selMcTable = mcTableXR.to_dataframe() 
    print('to_dataframe()')
    selTime = 600.
    times = selMcTable['time']
    selMcTable = selMcTable[times==selTime]
    #selMcTable = mcTable.copy()
    selMcTable['vel'] = -1. * selMcTable['vel']
    selMcTable['radii_mm'] = selMcTable['dia'] * 1e3 / 2.
    selMcTable['dia_mum'] = selMcTable['dia'] * 1e6 
    selMcTable['mTot_g'] = selMcTable['mTot'] * 1e3
    selMcTable['dia_cm'] = selMcTable['dia'] * 1e2
    if 'sPhi' not in selMcTable:
      selMcTable['sPhi'] = 1.0 # simply add sPhi = 1
    
    selMcTable = calcRhophys(selMcTable)
    selMcTable = calcRho(selMcTable)
    #selMcTable['sRho'] = 6.0e-3*mcTable.mTot/(np.pi*mcTable.dia**3*mcTable.sPhi**(-2+3*(mcTable.sPhi<1).astype(int)))
            
    return selMcTable

def kernel_estimate(R_SP_list,Rgrid,sigma0=0.62,weight=None,space='loge'): #taken and adapted from mo_output.f90
    """
    Calculate the kernel density estimate (kde) based on the super-particle list
    (adapted from McSnow's mo_output routines (f.e. write_distributions_meltdegree)
    Parameters
    ----------
    R_SP_list: list of the radii of the superparticle
    Rgrid: array of radii on which the kde is calculated
    sigma0: bandwidth prefactor of the kde (default value from Shima et al. (2009)
    weight: weight applied during integration (in this application the multiplicity)
    space: space in which the kde is applied (loge: logarithmix with base e, lin: linear space; D2: radii transformed by r_new=r**2) 
    """

    N_sp = len(R_SP_list) #number of superparticle
   
    #calculate bandwidth 
    sigmai = (sigma0 / N_sp**0.2) #ATTENTION:  this is defined **-1 in McSnow's mo_output.f90 (see also Shima et al (2009))

    #initialize number density
    N_R = np.zeros_like(Rgrid)
    
    expdiff_prefactor = 1./np.sqrt(2.*np.pi)/sigmai #calculate expdiff here to save computational time
    
    for i_rad,rad in enumerate(Rgrid):
        for i_SP,r in enumerate(R_SP_list): 
            
            #calculate weight
            if space=='loge':
                expdiff = expdiff_prefactor * np.exp(-0.5*((np.log(rad)-np.log(r))/sigmai)**2)
            elif space=='lin':
                expdiff = expdiff_prefactor * np.exp(-0.5*(((rad)-(r))/sigmai)**2)
            elif space=='D2':
                expdiff = expdiff_prefactor * np.exp(-0.5*(((rad)**2-(r)**2)/sigmai)**2)

            #integrate over each SP
            if weight is None: #if there is no weight
                N_R[i_rad] += expdiff #add sp%xi to this summation as in mo_output
            else:
                N_R[i_rad] += weight.iloc[i_SP]*expdiff #add sp%xi to this summation as in mo_output

            
    return N_R


def calcRho(mcTable):
    """
    Calculate the density of each super particles [g/cm^3].
    
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    
    Returns
    -------
    mcTable with an additional column for the density.
    The density is calculated separately for aspect ratio < 1
    and for aspect ratio >= 1.
    """
    
    # density calculation for different AR ranges
    mcTable['sRho'] = np.ones_like(mcTable['time'])*np.nan

    #calculaiton for AR < 1
    tmpTable = mcTable[mcTable['sPhi']<1].copy()
    tmpVol = (np.pi/6.) * (tmpTable['dia_cm'])**3 * tmpTable['sPhi']
    tmpRho = tmpTable['mTot_g']/tmpVol
    mcTable['sRho'].values[mcTable['sPhi']<1] = tmpRho

    # calculation for AR >= 1
    tmpTable = mcTable[mcTable['sPhi']>=1].copy()
    tmpVol = (np.pi/6.) * (tmpTable['dia_cm'])**3 / (tmpTable['sPhi'])**2
    tmpRho = (tmpTable['mTot_g'])/tmpVol
    mcTable['sRho'].values[mcTable['sPhi']>=1] = tmpRho
    
    return mcTable


# TODO this might return an xr.Dataset with wl as dimension instead
def creatRadarCols(mcTable, wls):
    """
    Create the Ze and KDP column
    
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    wls: wavelenght (iterable) [mm]
    
    Returns
    -------
    mcTable with a empty columns 'sZe*_*' 'sKDP_*' for 
    storing Ze_H and Ze_V and sKDP of one particle of a 
    given wavelength
    """
    
    for wl in wls:
    
        wlStr = '{:.2e}'.format(wl)
        mcTable['sZeH_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sZeV_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sKDP_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
     
        mcTable['sZeMultH_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sZeMultV_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sKDPMult_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan

    return mcTable
    
def calcRhophys(mcTable):
    """
    calculate the density of the particle, using the rime mass, ice mass, water mass,...
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    
    Returns
    -------
    mcTable with an additional column for the density.
    """
    rho_ice = 919.0
    rho_liq = 1000.0
    v_w_out = mcTable['sMrime']/rho_ice + mcTable['sMliqu']/rho_ice - mcTable['sVrime'] # do we have liquid water on the outside of the particle?
    v_tot = (mcTable['sMice'] + mcTable['sMmelt'])/rho_ice + mcTable['sVrime'] + v_w_out # total volume of the particle
    mcTable['sRho_tot'] = mcTable['mTot'] / v_tot
    return mcTable
    
    
