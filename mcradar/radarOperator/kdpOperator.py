# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import xarray as xr
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux

from mcradar.tableOperator import creatKdpCols

def calcKdpPropOneFreq(wl, radii, as_ratio, 
                       rho, elv, ndgs=2, canting=False, 
                       cantingStd=1, meanAngle=0):
    """
    Calculation of the KDP of one particle
    
    Parameters
    ----------
    wl: wavelength [mm] (single value)
    radii: radius [mm] of the particle (array[n])
    as_ratio: aspect ratio of the super particle (array[n])
    rho: density [g/mmˆ3] of the super particle (array[n])
    elv: elevation angle [°]
    ndgs: number of division points used to integrate over 
       the particle surface (default= 30 it is already high)
    canting: boolean (default = False)
    cantingStd: standard deviation of the canting angle [°] (default = 1)
    meanAngle: mean value of the canting angle [°] (default = 0)
    
    Returns
    -------
    kdp: calculated kdp from each particle (array[n])
    """
    
    scatterer = Scatterer(wavelength=wl)#, axis_ratio=1./as_ratio)
    scatterer.radius_type = Scatterer.RADIUS_MAXIMUM
    scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
    scatterer.ndgs = ndgs
    
    if canting==True: 
        scatterer.or_pdf = orientation.gaussian_pdf(std=cantingStd, mean=meanAngle)  
        #scatterer.orient = orientation.orient_averaged_adaptive
        scatterer.orient = orientation.orient_averaged_fixed
        
    # geometric parameters 
    scatterer.thet0 = 90. - elv
    scatterer.phi0 = 0.

    # geometric parameters 
    scatterer.thet = scatterer.thet0
    scatterer.phi = (scatterer.phi0) % 360. #KDP geometry

    sMat = np.ones_like(radii)*np.nan

    for i, radius in enumerate(radii):

        scatterer.axis_ratio = 1./as_ratio[i]
        scatterer.radius = radius
        scatterer.m = refractive.mi(wl, rho[i])
        S = scatterer.get_S()
        sMat[i] = (S[1,1]-S[0,0]).real
     
    kdp = 1e-3* (180.0/np.pi) * scatterer.wavelength *(sMat)
    return kdp


def calcParticleKDP(wls, elv, mcTable, ndgs=30):
    """
    Calculates kdp each superparticle from a given 
    distribution of super particles
    
    Parameters
    ----------
    wls: wavelenght [mm] (iterable)
    elv: elevation angle [°]
    mcTable: McSnow table returned from getMcSnowTable()
    ndgs: number of division points used to integrate over 
       the particle surface (default= 30 it is already high)

    Returns 
    -------
    mcTable including the kdp [°/km] of each super particle calculated
    for W band. The calculation is made separetely for aspect 
    ratio < 1 and >=1.
    """

    mcTable = creatKdpCols(mcTable, wls)
    
    ##calcualtion of the kdp for AR < 1
    meanAngle=0
    tmpTable = mcTable[mcTable['sPhi']<1].copy()

    radii_M1 = (tmpTable['radii'] * 1e3).values #[mm]
    as_ratio_M1 = tmpTable['sPhi'].values
    rho_M1 = tmpTable['sRho'].values #[g/cm^3]

    for wl in wls:
        
        kdp_M1 = calcKdpPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                   rho_M1, elv, ndgs=ndgs,
                                   canting=True, cantingStd=1., 
                                   meanAngle=meanAngle)

        wlStr = '{:.2e}'.format(wl)
        mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = kdp_M1
    
    
    ##calculation of the kdp for AR >= 1
    meanAngle=90
    tmpTable = mcTable[mcTable['sPhi']>=1].copy()

    radii_M1 = (tmpTable['radii'] * 1e3).values #[mm]
    as_ratio_M1 = tmpTable['sPhi'].values
    rho_M1 = tmpTable['sRho'].values #[g/cm^3]

    for wl in wls:
        
        kdp_M1 = calcKdpPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                   rho_M1, elv, ndgs=ndgs,
                                   canting=True, cantingStd=1., 
                                   meanAngle=meanAngle)

        wlStr = '{:.2e}'.format(wl)
        mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = kdp_M1
    
    return mcTable


def getIntKdp(wls, mcTable, centerHeight):
    """
    Calculates the integrated kdp of a distribution 
    of particles.
    
    Parameters
    ----------
    mcTable: McSnow table returned from calcParticleKDP()
    centerHeight: height of the center of the distribution 
    of particles
    
    Returns
    -------
    kdpXR: kdp calculated of a distribution of particles
    kdbXR: dims=(range)
    """
    
    tmpKdp = xr.Dataset()
        
    for wl in wls:
        
        wlStr = '{:.2e}'.format(wl)
        mcTable['sKDPMult_{0}'.format(wlStr)] = mcTable['sKDP_{0}'.format(wlStr)] * mcTable['sMult']
        
        kdpXR = xr.DataArray(mcTable['sKDPMult_{0}'.format(wlStr)].sum()[np.newaxis],
                             dims=('range'),
                             coords={'range':centerHeight[np.newaxis]},
                             name='kdpInt_{0}'.format(wlStr))
        
        tmpKdp = xr.merge([tmpKdp, kdpXR])
    
    return tmpKdp


