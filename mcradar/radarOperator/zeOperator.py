# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux

from mcradar.tableOperator import creatZeCols


def calcScatPropOneFreq(wl, radii, as_ratio, 
                        rho, elv, ndgs=30,
                        canting=False, cantingStd=1, 
                        meanAngle=0):
    """
    Calculates the Ze of one particle
    
    Parameters
    ----------
    wl: wavelenght [mm] (single value)
    radii: radius [mm] of the particle (array[n])
    as_ratio: aspect ratio of the super particle (array[n])
    rho: density [g/mmˆ3] of the super particle (array[n])
    elv: elevation angle [°]
    ndgs: division points used to integrate over the particle surface
    canting: boolean (default = False)
    cantingStd: standard deviation of the canting angle [°] (default = 1)
    meanAngle: mean value of the canting angle [°] (default = 0)
    
    Returns
    -------
    reflect: super particle horizontal reflectivity[mm^6/m^3] (array[n])
    reflect_v: super particle vertical reflectivity[mm^6/m^3] (array[n])
    refIndex: refractive index from each super particle (array[n])
    """
    
    #---pyTmatrix setup
    # initialize a scatterer object
    scatterer = Scatterer(wavelength=wl)
    scatterer.radius_type = Scatterer.RADIUS_MAXIMUM
    scatterer.ndgs = ndgs
    scatterer.ddelta = 1e-6

    if canting==True: 
        scatterer.or_pdf = orientation.gaussian_pdf(std=cantingStd, mean=meanAngle)  
#         scatterer.orient = orientation.orient_averaged_adaptive
        scatterer.orient = orientation.orient_averaged_fixed
    
    # geometric parameters 
    scatterer.thet0 = 90. - elv
    scatterer.phi0 = 0.
    
    # geometric parameters 
    scatterer.thet = 180. - scatterer.thet0
    scatterer.phi = (180. + scatterer.phi0) % 360.

    refIndex = np.ones_like(radii, np.complex128)*np.nan
    reflect = np.ones_like(radii)*np.nan
    reflect_v = np.ones_like(radii)*np.nan
    
    for i, radius in enumerate(radii):
        
        scatterer.radius = radius
        scatterer.axis_ratio = 1./as_ratio[i]
        scatterer.m = refractive.mi(wl, rho[i])
        refIndex[i] = refractive.mi(wl, rho[i])
        reflect[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, True)
        reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, False)
        
    del scatterer
    return reflect, reflect_v, refIndex

    
def calcParticleZe(wls, elv, mcTable, ndgs=30):#zeOperator
    """
    Calculates the horizontal and vertical reflectivity of 
    each superparticle from a given distribution of super 
    particles
    
    Parameters
    ----------
    wls: wavelenght [mm] (iterable)
    elv: elevation angle [°]
    mcTable: McSnow table returned from getMcSnowTable()
    ndgs: division points used to integrate over the particle surface

    Returns 
    -------
    mcTable including the horizontal and vertical reflectivity
    of each super particle calculated for X, Ka and W band. The
    calculation is made separetely for aspect ratio < 1 and >=1.
    """
    
    #calling the function to create Ze columns
    mcTable = creatZeCols(mcTable, wls)
    #mcTable = mcTable.sort_values('dia')
    
    ##calculation of the reflectivity for AR < 1
    tmpTable = mcTable[mcTable['sPhi']<1].copy()

    #particle properties
    canting = True
    meanAngle=0
    cantingStd=1
    
    radii_M1 = (tmpTable['radii'] * 1e3).values #[mm]
    as_ratio_M1 = tmpTable['sPhi'].values
    rho_M1 = tmpTable['sRho'].values #[g/cm^3]

    for wl in wls:
                
        reflect_h,  reflect_v, refInd = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                                            rho_M1, elv, canting=canting, 
                                                            cantingStd=cantingStd, 
                                                            meanAngle=meanAngle, ndgs=ndgs)
        wlStr = '{:.2e}'.format(wl)
        mcTable['sZeH_{0}'.format(wlStr)][mcTable['sPhi']<1] = reflect_h
        mcTable['sZeV_{0}'.format(wlStr)][mcTable['sPhi']<1] = reflect_v

    ##calculation of the reflectivity for AR >= 1
    tmpTable = mcTable[mcTable['sPhi']>=1].copy()
    
    #particle properties
    canting=True
    meanAngle=90
    cantingStd=1
    
    radii_M1 = (tmpTable['radii'] * 1e3).values #[mm]
    as_ratio_M1 = tmpTable['sPhi'].values
    rho_M1 = tmpTable['sRho'].values #[g/cm^3]

    for wl in wls:
    
        reflect_h,  reflect_v, refInd = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                                            rho_M1, elv, canting=canting, 
                                                            cantingStd=cantingStd, 
                                                            meanAngle=meanAngle, ndgs=ndgs)
        wlStr = '{:.2e}'.format(wl)
        mcTable['sZeH_{0}'.format(wlStr)][mcTable['sPhi']>=1] = reflect_h
        mcTable['sZeV_{0}'.format(wlStr)][mcTable['sPhi']>=1] = reflect_v

    return mcTable




