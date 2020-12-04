# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux

from mcradar.tableOperator import creatRadarCols


def calcScatPropOneFreq(wl, radii, as_ratio, 
                        rho, elv, ndgs=30,
                        canting=False, cantingStd=1, 
                        meanAngle=0):
    """
    Calculates the Ze at H and V polarization, Kdp for one wavelength
    TODO: LDR???
    
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
    kdp: calculated kdp from each particle (array[n])
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
    
    # geometric parameters - incident direction
    scatterer.thet0 = 90. - elv
    scatterer.phi0 = 0.
    
    # parameters for backscattering
    refIndex = np.ones_like(radii, np.complex128)*np.nan
    reflect_h = np.ones_like(radii)*np.nan
    reflect_v = np.ones_like(radii)*np.nan

    # S matrix for Kdp
    sMat = np.ones_like(radii)*np.nan

    for i, radius in enumerate(radii):
        # scattering geometry backward
        scatterer.thet = 180. - scatterer.thet0
        scatterer.phi = (180. + scatterer.phi0) % 360.
        scatterer.radius = radius
        scatterer.axis_ratio = 1./as_ratio[i]
        scatterer.m = refractive.mi(wl, rho[i])
        refIndex[i] = refractive.mi(wl, rho[i])
        reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, True)
        reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, False)

        # scattering geometry forward
        scatterer.thet = scatterer.thet0
        scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
        S = scatterer.get_S()
        sMat[i] = (S[1,1]-S[0,0]).real
    kdp = 1e-3* (180.0/np.pi)*scatterer.wavelength*sMat

    del scatterer # TODO: Evaluate the chance to have one Scatterer object already initiated instead of having it locally
    return reflect_h, reflect_v, refIndex, kdp

    
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
    
    #calling the function to create output columns
    mcTable = creatRadarCols(mcTable, wls)

    ##calculation of the reflectivity for AR < 1
    tmpTable = mcTable[mcTable['sPhi']<1].copy()

    #particle properties
    canting = True
    meanAngle=0
    cantingStd=1
    
    radii_M1 = tmpTable['radii_mm'].values #[mm]
    as_ratio_M1 = tmpTable['sPhi'].values
    rho_M1 = tmpTable['sRho'].values #[g/cm^3]

    for wl in wls:
                
        singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                         rho_M1, elv, canting=canting, 
                                         cantingStd=cantingStd, 
                                         meanAngle=meanAngle, ndgs=ndgs)
        reflect_h,  reflect_v, refInd, kdp_M1 = singleScat
        wlStr = '{:.2e}'.format(wl)
        mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = reflect_h
        mcTable['sZeV_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = reflect_v
        mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = kdp_M1


    ##calculation of the reflectivity for AR >= 1
    tmpTable = mcTable[mcTable['sPhi']>=1].copy()
    
    #particle properties
    canting=True
    meanAngle=90
    cantingStd=1
    
    radii_M1 = (tmpTable['radii_mm']).values #[mm]
    as_ratio_M1 = tmpTable['sPhi'].values
    rho_M1 = tmpTable['sRho'].values #[g/cm^3]

    for wl in wls:
    
        singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                         rho_M1, elv, canting=canting, 
                                         cantingStd=cantingStd, 
                                         meanAngle=meanAngle, ndgs=ndgs)
        reflect_h,  reflect_v, refInd, kdp_M1 = singleScat
        wlStr = '{:.2e}'.format(wl)
        mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = reflect_h
        mcTable['sZeV_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = reflect_v
        mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = kdp_M1

    return mcTable


