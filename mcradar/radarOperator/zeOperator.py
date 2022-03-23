# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess
import numpy as np
import xarray as xr
from glob import glob
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux

from mcradar.tableOperator import creatRadarCols
import matplotlib.pyplot as plt
# TODO: this function should deal with the LUTs
def calcScatPropOneFreq(wl, radii, as_ratio, 
                        rho, elv, ndgs=30,
                        canting=False, cantingStd=1, 
                        meanAngle=0, safeTmatrix=False):
    """
    Calculates the Ze at H and V polarization, Kdp for one wavelength
    TODO: LDR???
    
    Parameters
    ----------
    wl: wavelength [mm] (single value)
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
    reflect_h: super particle horizontal reflectivity[mm^6/m^3] (array[n])
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

    for i, radius in enumerate(radii[::5]): #TODO remove [::5]
        # A quick function to save the distribution of values used in the test
        #with open('/home/dori/table_McRadar.txt', 'a') as f:
        #    f.write('{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f}\n'.format(wl, elv,
        #                                                                 meanAngle,
        #                                                                 cantingStd,
        #                                                                 radius,
        #                                                                 rho[i],
        #                                                                 as_ratio[i]))
        # scattering geometry backward
        # radius = 100.0 # just a test to force nans

        scatterer.thet = 180. - scatterer.thet0
        scatterer.phi = (180. + scatterer.phi0) % 360.
        scatterer.radius = radius
        scatterer.axis_ratio = 1./as_ratio[i]
        scatterer.m = refractive.mi(wl, rho[i])
        refIndex[i] = refractive.mi(wl, rho[i])

        if safeTmatrix:
            inputs = [str(scatterer.radius),
                      str(scatterer.wavelength),
                      str(scatterer.m),
                      str(scatterer.axis_ratio),
                      str(int(canting)),
                      str(cantingStd),
                      str(meanAngle),
                      str(ndgs),
                      str(scatterer.thet0),
                      str(scatterer.phi0)]
            arguments = ' '.join(inputs)
            a = subprocess.run(['spheroidMcRadar'] + inputs, # this script should be installed by McRadar
                               capture_output=True)
            # print(str(a))
            try:
                back_hh, back_vv, sMatrix, _ = str(a.stdout).split('Results ')[-1].split()
                back_hh = float(back_hh)
                back_vv = float(back_vv)
                sMatrix = float(sMatrix)
            except:
                back_hh = np.nan
                back_vv = np.nan
                sMatrix = np.nan
            # print(back_hh, radar.radar_xsect(scatterer, True))
            # print(back_vv, radar.radar_xsect(scatterer, False))
            reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * back_hh # radar.radar_xsect(scatterer, True)  # Kwsqrt is not correct by default at every frequency
            reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * back_vv # radar.radar_xsect(scatterer, False)

            # scattering geometry forward
            # scatterer.thet = scatterer.thet0
            # scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
            # S = scatterer.get_S()
            sMat[i] = sMatrix # (S[1,1]-S[0,0]).real
            # print(sMatrix, sMat[i])
            # print(sMatrix)
        else:

            reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, True)  # Kwsqrt is not correct by default at every frequency
            reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, False)

            # scattering geometry forward
            scatterer.thet = scatterer.thet0
            scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
            S = scatterer.get_S()
            sMat[i] = (S[1,1]-S[0,0]).real

    kdp = 1e-3* (180.0/np.pi)*scatterer.wavelength*sMat

    del scatterer # TODO: Evaluate the chance to have one Scatterer object already initiated instead of having it locally
    return reflect_h, reflect_v, refIndex, kdp


def radarScat(sp, wl, K2=0.93):
    """
    Calculates the single scattering radar quantities from the matrix values
    Parameters
    ----------
    sp: dataArray [n] superparticles containing backscattering matrix 
            and forward amplitude matrix information needed to compute
            spectral radar quantities
    wl: wavelength [mm]
    K2: Rayleigh dielectric factor |(m^2-1)/(m^2+2)|^2

    Returns
    -------
    reflect_h: super particle horizontal reflectivity[mm^6/m^3] (array[n])
    reflect_v: super particle vertical reflectivity[mm^6/m^3] (array[n])
    kdp: calculated kdp from each particle (array[n])
    ldr_h: linear depolarization ratio horizontal (array[n])
    rho_hv: correlation coefficient (array[n])
    """
    prefactor = 2*np.pi*wl**4/(np.pi**5*K2)
    #print(sp.Z11.values)
    #quit()
    reflect_hh = prefactor*(sp.Z11 - sp.Z12 - sp.Z21 + sp.Z22).values
    reflect_vv = prefactor*(sp.Z11 + sp.Z12 + sp.Z21 + sp.Z22).values
    kdp = 1e-3*(180.0/np.pi)*wl*sp.S22r_S11r.values

    reflect_hv = prefactor*(sp.Z11 - sp.Z12 + sp.Z21 - sp.Z22).values
    #reflect_vh = prefactor*(sp.Z11 + sp.Z12 - sp.Z21 - sp.Z22).values
    ldr_h = reflect_hv/reflect_hh
               
    # delta_hv np.arctan2(Z[2,3] - Z[3,2], -Z[2,2] - Z[3,3])
    #a = (Z[2,2] + Z[3,3])**2 + (Z[3,2] - Z[2,3])**2
    #b = (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    #c = (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])
    #rho_hv np.sqrt(a / (b*c))
    rho_hv = np.nan*np.ones_like(reflect_hh) # disable rho_hv for now
    #Ah = 4.343e-3 * 2 * scatterer.wavelength * sp.S22i.values # attenuation horizontal polarization
    #Av = 4.343e-3 * 2 * scatterer.wavelength * sp.S11i.values # attenuation vertical polarization

    return reflect_hh, reflect_vv, kdp, ldr_h, rho_hv

    
def calcParticleZe(wls, elv, mcTable, ndgs=30,
                   scatSet={'mode':'full', 'safeTmatrix':False}, K2=0.93):#zeOperator
    """
    Calculates the horizontal and vertical reflectivity of 
    each superparticle from a given distribution of super 
    particles,in this case I just quickly wanted to change the function to deal with Monomers with the DDA LUT and use Tmatrix for the aggregates
    
    Parameters
    ----------
    wls: wavelength [mm] (iterable)
    elv: elevation angle [°] # TODO: maybe also this can become iterable
    mcTable: McSnow table returned from getMcSnowTable()
    ndgs: division points used to integrate over the particle surface
    scatSet: type of scattering calculations to use, choose between full, table, wisdom, SSRGA, Rayleigh or SSRGA-Rayleigh
    Returns 
    -------
    mcTable including the horizontal and vertical reflectivity
    of each super particle calculated for X, Ka and W band. The
    calculation is made separetely for aspect ratio < 1 and >=1.
    Kdp is also included. TODO spectral ldr and rho_hv
    """
    
    #calling the function to create output columns
    mcTable = creatRadarCols(mcTable, wls)
    #print('mcTable has ', len(mcTable))

    if scatSet['mode'] == 'full':
        print('Full mode Tmatrix calculation')
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
                                             meanAngle=meanAngle, ndgs=ndgs,
                                             safeTmatrix=scatSet['safeTmatrix'])
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
                                             meanAngle=meanAngle, ndgs=ndgs,
                                             safeTmatrix=scatSet['safeTmatrix'])
            reflect_h,  reflect_v, refInd, kdp_M1 = singleScat
            wlStr = '{:.2e}'.format(wl)
            mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = reflect_h
            mcTable['sZeV_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = reflect_v
            mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = kdp_M1

    elif scatSet['mode'] == 'SSRGA':
        print('using SSRGA scattering table for all particles, elevation is set to 90')
        lut = xr.open_dataset(scatSet['lutFile'])
        for wl in wls:
            points = lut.sel(wavelength=wl*1e-3, elevation=90.0, # sofar: elevation can only be 90, we need more SSRGA calculations for other elevation
                             size = xr.DataArray(mcTable['radii_mm'].values, dims='points'),
                             method='nearest')
            ssCbck = points.Cbck.values*1e6 # Tmatrix output is in mm, so here we also have to use ssCbck in mm
            prefactor = wl**4/(np.pi**5*K2) # other prefactor: 2*pi*...
            wlStr = '{:.2e}'.format(wl)
            mcTable['sZeH_{0}'.format(wlStr)] = prefactor*ssCbck
            
    elif scatSet['mode'] == 'Rayleigh':
        print('using Rayleigh approximation for all particles, only elevation 90 so far')
        for wl in wls:
            '''
            # rayleigh approximation taken from Stefans ssrg_general folder
            # calculate equivalent radius from equivalent mass
            re = ((3 * mcTable['mTot']) / (4 * mcTable['sRhoIce'] * np.pi)) ** (1/3) *1e3
            X = 4*np.pi*re/wl # calculate size parameter
            qbck_h = 4*X**4*K2 # calculate backscattering efficiency
            cbck_h = qbck_h  * re**2 * np.pi/((2*np.pi)**2) #need to divide by (2*pi)^2 to get same as ssrga
            '''
            wlStr = '{:.2e}'.format(wl)
            prefactor = wl**4/(np.pi**5*K2)
            
            # rayleigh approximation according to Jussi Leinonens diss:             
            k = 2 * np.pi / (wl*1e-3)
            sigma = 4*np.pi*np.abs(3*k**2/(4*np.pi)*np.sqrt(K2)*(mcTable['mTot']/mcTable['sRho_tot']))**2 *1e6 /np.pi #need to divide by pi to get same as ssrga, need to multiply by 1e6 to get mm^3
            mcTable['sZeH_{0}'.format(wlStr)] = prefactor * sigma
    elif scatSet['mode'] == 'SSRGA-Rayleigh':
        print('using SSRGA for aggregates and Rayleigh approximation for crystals, only elevation 90')
        mcTableAgg = mcTable[(mcTable['sNmono']>1)].copy() # only aggregates
        mcTableCry = mcTable[(mcTable['sNmono']==1)].copy() # only monomers
        lut = xr.open_dataset(scatSet['lutFile']) # SSRGA LUT
        for wl in wls:
            wlStr = '{:.2e}'.format(wl)
            prefactor = wl**4/(np.pi**5*K2)            
            # rayleigh approximation for crystal:             
            k = 2 * np.pi / (wl*1e-3)
            if len(mcTableCry['mTot']) > 0:
              ssCbck = 4*np.pi*np.abs(3*k**2/(4*np.pi)*np.sqrt(K2)*(mcTableCry['mTot']/mcTableCry['sRho_tot']))**2 *1e6 /np.pi #need to divide by pi to get same as ssrga, need to multiply by 1e6 to get mm^3
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']==1] = prefactor * ssCbck
            else:
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']==1] = np.nan
            # ssrga for aggregates:
            points = lut.sel(wavelength=wl*1e-3, elevation=90.0, # sofar: elevation can only be 90, we need more SSRGA calculations for other elevation
                             size = xr.DataArray(mcTableAgg['radii_mm'].values, dims='points'),
                             method='nearest')
            
            if len(points.Cbck)>0: # only if we have aggregates this works. Otherwise we need to write nan here
              ssCbck = points.Cbck.values*1e6 # in mm^3
              prefactor = wl**4/(np.pi**5*K2) 
              wlStr = '{:.2e}'.format(wl)
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']>1] = prefactor*ssCbck
            else:
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']>1] = np.nan
    elif len(mcTable): # interpolation fails if no selection is possible
        elvSel = scatSet['lutElev'][np.argmin(np.abs(np.array(scatSet['lutElev'])-elv))]
        print('elevation ', elv,'lut elevation ', elvSel)
        for wl in wls:
            f = 299792458e3/wl
            freSel = scatSet['lutFreq'][np.argmin(np.abs(np.array(scatSet['lutFreq'])-f))]
            print('frequency ', f/1.e9, 'lut frequency ', freSel/1.e9)
            dataset_filename = scatSet['lutPath'] + 'testLUT_{:3.1f}e9Hz_{:d}.nc'.format(freSel/1e9, int(elvSel)) 
            lut = xr.open_dataset(dataset_filename)#.sel(wavelength=wl,
                                                   #     elevation=elv,
                                                   #     canting=1.0,
                                                   #     method='nearest')

            points = lut.sel(wavelength=wl, elevation=elv, canting=1.0,
                             size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
                             aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
                             density=xr.DataArray(mcTable['sRho'].values, dims='points'),
                             method='nearest')

            reflect_h,  reflect_v, kdp_M1, ldr, rho_hv = radarScat(points, wl)
            wlStr = '{:.2e}'.format(wl)
            mcTable['sZeH_{0}'.format(wlStr)].values = reflect_h
            mcTable['sZeV_{0}'.format(wlStr)].values = reflect_v
            mcTable['sKDP_{0}'.format(wlStr)].values = kdp_M1

            if scatSet['mode'] == 'table':
                print('fast LUT mode')

            elif scatSet['mode'] == 'wisdom':            
                print('less fast cache adaptive mode')

    return mcTable


