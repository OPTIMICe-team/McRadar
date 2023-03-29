# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess
import numpy as np
import xarray as xr
from glob import glob
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux
from scipy import constants
from scipy.optimize import curve_fit
from mcradar.tableOperator import creatRadarCols
import matplotlib.pyplot as plt
import pandas as pd
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
    Z11Mat = np.ones_like(radii)*np.nan
    Z12Mat = np.ones_like(radii)*np.nan
    Z21Mat = np.ones_like(radii)*np.nan
    Z22Mat = np.ones_like(radii)*np.nan
    Z33Mat = np.ones_like(radii)*np.nan
    Z44Mat = np.ones_like(radii)*np.nan
    S11iMat = np.ones_like(radii)*np.nan
    S22iMat = np.ones_like(radii)*np.nan
    for i, radius in enumerate(radii): #TODO remove [::5]
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
                back_hh, back_vv, sMatrix, Z11, Z12, Z21, Z22, Z33, Z44, S11i, S22i, _ = str(a.stdout).split('Results ')[-1].split()
                back_hh = float(back_hh)
                back_vv = float(back_vv)
                sMatrix = float(sMatrix)
                Z11 = float(Z11)
                Z12 = float(Z12)
                Z21 = float(Z21)
                Z22 = float(Z22)
                Z33 = float(Z33)
                Z44 = float(Z44)
                S11i = float(S11i)
                S22i = float(S22i)
            except:
                back_hh = np.nan
                back_vv = np.nan
                sMatrix = np.nan
                Z11 = np.nan
                Z12 = np.nan
                Z21 = np.nan
                Z22 = np.nan
                Z33 = np.nan
                Z44 = np.nan
                S11i = np.nan
                S22i = np.nan
            # print(back_hh, radar.radar_xsect(scatterer, True))
            # print(back_vv, radar.radar_xsect(scatterer, False))
            reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * back_hh # radar.radar_xsect(scatterer, True)  # Kwsqrt is not correct by default at every frequency
            reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * back_vv # radar.radar_xsect(scatterer, False)

            # scattering geometry forward
            # scatterer.thet = scatterer.thet0
            # scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
            # S = scatterer.get_S()
            sMat[i] = sMatrix # (S[1,1]-S[0,0]).real
            Z11Mat[i] = Z11
            Z12Mat[i] = Z12
            Z21Mat[i] = Z21
            Z22Mat[i] = Z22
            Z33Mat[i] = Z33
            Z44Mat[i] = Z44
            S11iMat[i] = S11i
            S22iMat[i] = S22i
            # print(sMatrix, sMat[i])
            # print(sMatrix)
        else:

            reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, True)  # Kwsqrt is not correct by default at every frequency
            reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, False)

            # scattering geometry forward
            scatterer.thet = scatterer.thet0
            scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
            S = scatterer.get_S()
            Z = scatterer.get_Z()
            sMat[i] = (S[1,1]-S[0,0]).real
            Z11Mat[i] = Z[0,0]
            Z12Mat[i] = Z[0,1]
            Z21Mat[i] = Z[1,0]
            Z22Mat[i] = Z[1,1]
            Z33Mat[i] = Z[2,2]
            Z44Mat[i] = Z[3,3]
            S11iMat[i] = S[0,0].imag
            S22iMat[i] = S[1,1].imag
            
    kdp = 1e-3* (180.0/np.pi)*scatterer.wavelength*sMat

    del scatterer # TODO: Evaluate the chance to have one Scatterer object already initiated instead of having it locally
    return reflect_h, reflect_v, refIndex, kdp, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat


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
    #reflect_hh = prefactor*(sp.Z11 - sp.Z12 - sp.Z21 + sp.Z22).values #TODO why is it here the other way around compared to what Davide has in his notebooks???
    #reflect_vv = prefactor*(sp.Z11 + sp.Z12 + sp.Z21 + sp.Z22).values
    reflect_hh = prefactor*(sp.Z11+sp.Z22+sp.Z12+sp.Z21).values
    reflect_vv = prefactor*(sp.Z11+sp.Z22-sp.Z12-sp.Z21).values
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

def rational3d(x,y,z,a,b):
    if len(a)==17 and len(b)==9:
        z = rat3dp26(x,y,z,a,b)
    elif len(a)==13 and len(b)==12:
        z = rat3dp25(x,y,z,a,b)
    elif len(a)==13 and len(b)==9:
        z = rat3dp22(x,y,z,a,b)
    elif len(a)==10 and len(b)==9:
        z = rat3dp19(x,y,z,a,b)
    elif len(a)==9 and len(b)==8:
        z = rat3dp17x(x,y,z,a,b)
    elif len(a)==8 and len(b)==7:
        z = rat3dp15(x,y,z,a,b)
    elif len(a)==7 and len(b)==6:
        z = rat3dp13(x,y,z,a,b)
    elif len(a)==4 and len(b)==3:
        z = rat3dp7(x,y,z,a,b)
    return z

def rat3dp26(x, y, z, a, b):
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*y*z+a[8]*z*z+a[9]*x*z \
            +a[10]*x*x*x+a[11]*y*y*y+a[12]*z*z*z++a[13]*x*x*z+a[14]*y*y*z+a[15]*x*z*z+a[16]*y*z*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*y*z+b[7]*z*z+b[8]*x*z
    return p/q

def rat3dp25(x, y, z, a, b):
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*y*z+a[8]*z*z+a[9]*x*z+a[10]*x*x*x+a[11]*y*y*y+a[12]*z*z*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*y*z+b[7]*z*z+b[8]*x*z+b[9]*x*x*x+b[10]*y*y*y+b[11]*z*z*z
    return p/q

def rat3dp22(x, y, z, a, b):
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*y*z+a[8]*z*z+a[9]*x*z+a[10]*x*x*x+a[11]*y*y*y+a[12]*z*z*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*y*z+b[7]*z*z+b[8]*x*z
    return p/q

def rat3dp19(x, y, z, a, b):
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*y*z+a[8]*z*z+a[9]*x*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*y*z+b[7]*z*z+b[8]*x*z
    return p/q

def rat3dp17x(x, y, z, a, b): # ohne den gemischten Term in y*z
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*z*z+a[8]*x*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*z*z+b[7]*x*z
    return p/q

def rat3dp17y(x, y, z, a, b): # ohne den gemischten Term in x*z
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*z*z+a[8]*y*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*z*z+b[7]*y*z
    return p/q

def rat3dp15(x, y, z, a, b): # ohne beide gemischte Terme in x*z und y*z
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*z*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*z*z
    return p/q

def rat3dp13(x, y, z, a, b): # ohne beide gemischte Terme in x*z und y*z
    p = a[0]+a[1]*x+a[2]*y+a[3]*z+a[4]*x*x+a[5]*y*y+a[6]*z*z #+a[5]*x*y
    q = 1.0+b[0]*x+b[1]*y+b[2]*z+b[3]*x*x+b[4]*y*y+b[5]*z*z #b[4]*x*y
    return p/q
def rat3dp7(x, y, z, a, b): # ohne beide gemischte Terme in x*z und y*z
    p = a[0]+a[1]*x+a[2]*y+a[3]*z #+a[4]*x*x+a[5]*x*y+a[6]*y*y+a[7]*z*z
    q = 1.0+b[0]*x+b[1]*y+b[2]*z #+b[3]*x*x+b[4]*x*y+b[5]*y*y+b[6]*z*z
    return p/q

def get_ab_from_params(p):
    n = len(p)
    if n == 26:
        a,b = p[0:17],p[17:26]
    elif n == 25:
        a,b = p[0:13],p[13:25]
    elif n == 22:
        a,b = p[0:13],p[13:22]
    elif n == 19:
        a,b = p[0:10],p[10:19]
    elif n == 17:
        a,b = p[0:9],p[9:17]
    elif n == 15:
        a,b = p[0:8],p[8:15]
    elif n == 13:
        a,b = p[0:7],p[7:13]
    elif n == 11:
        a,b = p[0:6],p[6:11]
    elif n == 7:
        a,b = p[0:4],p[4:7]
    
    return a,b

def _rational(M, *args):
	# This is the callable function that is passed to curve_fit. M is a (3,N) array
	# where N is the total number of data points in Z, which will be raveled
	# to one dimension. The rewrite as a one-dimensional function is necessary for using 
	# curve_fit, which supports only one-dimensional functions.

    x,y,z = M
    a,b = get_ab_from_params(args)
    var = rational3d(x,y,z,a,b)
    return var
    
def calcParticleZe(wls, elvs, mcTable, ndgs=30,
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
    #mcTable = creatRadarCols(mcTable, wls)
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
        rho_M1 = tmpTable['sRho_tot_g'].values #[g/cm^3]

        for wl in wls:
                    
            singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                             rho_M1, elv, canting=canting, 
                                             cantingStd=cantingStd, 
                                             meanAngle=meanAngle, ndgs=ndgs,
                                             safeTmatrix=scatSet['safeTmatrix'])
            reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
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
        rho_M1 = tmpTable['sRho_tot'].values #[g/cm^3]

        for wl in wls:
        
            singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                             rho_M1, elv, canting=canting, 
                                             cantingStd=cantingStd, 
                                             meanAngle=meanAngle, ndgs=ndgs,
                                             safeTmatrix=scatSet['safeTmatrix'])
            reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
            wlStr = '{:.2e}'.format(wl)
            mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = reflect_h
            mcTable['sZeV_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = reflect_v
            mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']>=1] = kdp_M1

    elif scatSet['mode'] == 'SSRGA':
        print('using SSRGA scattering table for all particles, elevation is set to 90')
        lut = xr.open_dataset(scatSet['lutFile'])
        for wl in wls:
            freq = (constants.c / wl*1e3)
            #print(freq)
            #quit()
            mcTableAgg = mcTable[(mcTable['sNmono']>1)].copy()
            points = lut.sel(frequency=freq, temperature=270.0, elevation=elv, 
                             size = xr.DataArray(mcTableAgg['dia'].values, dims='points'),
                             method='nearest')
            #points = lut.sel(wavelength=wl*1e-3, elevation=90.0, # sofar: elevation can only be 90, we need more SSRGA calculations for other elevation
            #                 size = xr.DataArray(mcTable['dia'].values, dims='points'),
            #                 method='nearest')
            ssCbck = points.Cbck.values#*1e6 # Tmatrix output is in mm, so here we also have to use ssCbck in mm
            
            prefactor = wl**4/(np.pi**5*K2) # other prefactor: 2*pi*...
            wlStr = '{:.2e}'.format(wl)
            mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']>1] = prefactor*ssCbck * 1e18
            mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']==1] = np.nan
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
            prefactor = (wl*1e-3)**4/(np.pi**5*K2)
            
            # rayleigh approximation according to Jussi Leinonens diss:             
            k = 2 * np.pi / (wl*1e-3)
            sigma = 4*np.pi*np.abs(3*k**2/(4*np.pi)*np.sqrt(K2)*(mcTable['mTot']/mcTable['sRho_tot']))**2/np.pi #need to divide by pi to get same as ssrga
            mcTable['sZeH_{0}'.format(wlStr)] = prefactor * sigma * 1e18 # 1e18 to convert to mm6/m3
    elif scatSet['mode'] == 'SSRGA-Rayleigh':
        print('using SSRGA for aggregates and Rayleigh approximation for crystals')
        
        mcTableAgg = mcTable[(mcTable['sNmono']>1)].copy() # only aggregates
        mcTableCry = mcTable[(mcTable['sNmono']==1)].copy() # only monomers
        lut = xr.open_dataset(scatSet['lutFile']) # SSRGA LUT
        for wl in wls:
            wlStr = '{:.2e}'.format(wl) # wl here is in mm!!
            
            wl_m = wl*1e-3
            prefactor = wl_m**4/(np.pi**5*K2)            
            # rayleigh approximation for crystal:             
            k = 2 * np.pi / (wl_m)
            if len(mcTableCry['mTot']) > 0:
              ssCbck = 4*np.pi*np.abs(3*k**2/(4*np.pi)*np.sqrt(K2)*(mcTableCry['mTot']/mcTableCry['sRho_tot']))**2/np.pi #need to divide by pi to get same as ssrga
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']==1] = prefactor * ssCbck * 1e18 # 1e18 to convert to mm6/m3
            else:
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']==1] = np.nan
            # ssrga for aggregates:
            
            freq = constants.c / wl_m
            #if len(mcTableAgg['mTot']) > 0:
                
                #quit()
            points = lut.sel(frequency=freq, temperature=270.0, elevation=elv, 
                             size = xr.DataArray(mcTableAgg['dia'].values, dims='points'),
                             method='nearest') # select nearest particle properties.
            
            if len(points.Cbck)>0: # only if we have aggregates this works. Otherwise we need to write nan here
              ssCbck = points.Cbck.values # in mm^3 
              
              #prefactor = (wl_m)**4/(np.pi**5*K2) 
              wlStr = '{:.2e}'.format(wl)
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']>1] = prefactor*ssCbck * 1e18 # 1e18 to convert to mm6/m3
            else:
              mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sNmono']>1] = np.nan
    elif scatSet['mode'] == 'DDA_old':
        lut = xr.open_dataset(scatSet['lutFile'])
        for wl in wls:
            wlStr = '{:.2e}'.format(wl)
            freq = (constants.c / wl*1e3)
            prefactor = 2*np.pi*wl**4/(np.pi**5*K2)
            points = lut.sel(frequency=freq, elevation=elv, 
                             Dmax = xr.DataArray(mcTable['dia'].values, dims='points'),
                             method='nearest')
            if len(points.cbck_hh)>0: # only if we have aggregates this works. Otherwise we need to write nan here
              ssCbckhh = points.cbck_hh.values#*1e6 # in mm^3 
              #print(ssCbckhh)
              ssCbckvv = points.cbck_vv.values#*1e6
              KDP = points.kdp.values
              mcTable['sZeH_{0}'.format(wlStr)] = prefactor * ssCbckhh * 1e18 # 1e18 to convert to mm6/m3
              mcTable['sZeV_{0}'.format(wlStr)] = prefactor * ssCbckvv * 1e18 # 1e18 to convert to mm6/m3
              mcTable['sKDP_{0}'.format(wlStr)] = KDP
            else:
              mcTable['sZeH_{0}'.format(wlStr)] = np.nan
              mcTable['sZeV_{0}'.format(wlStr)] = np.nan
              mcTable['sKDP_{0}'.format(wlStr)] = np.nan
    
    elif scatSet['mode'] == 'DDA':
        #-- this option uses the DDA scattering tables.
        #calculation of the reflectivity for AR < 1
        #print('DDA selected, only possible for plate-like particles at the moment. Also, scattering of aggregates gets calulated with SSRGA')
        # different DDA LUT for monomers and Aggregates. Sofar only for dendritic particles. 
        mcTabledendrite = mcTable.where(mcTable['sPhi']<1,drop=True) # select only plates
        mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True) # select only aggregates
        for wl in wls:
            wlStr = '{:.2e}'.format(wl)
            f = 299792458e3/wl
            for elv in elvs:

                if len(mcTabledendrite.sPhi)>0: # only possible if we have plate-like particles
                    elvSelMono = scatSet['lutElevMono'][np.argmin(np.abs(np.array(scatSet['lutElevMono'])-elv))] # get correct elevation of LUT
                    freSel = scatSet['lutFreqMono'][np.argmin(np.abs(np.array(scatSet['lutFreqMono'])-f/1e9))] # get correct frequency of LUT
                    freSel = str(freSel).ljust(6,'0')#
                    dataset_filename = scatSet['lutPath'] + 'DDA_LUT_dendrites_freq{}_elv{:d}.nc'.format(freSel, int(elvSelMono)) # get filename of LUT
                    lut = xr.open_dataset(dataset_filename)
                    lutsel = lut.sel(wavelength=wl*1e-3, elevation=elv,method='nearest') # select nearest wl and elevation TODO: make wavelength in mm
                    points = lutsel.interp(Dmax=xr.DataArray(mcTabledendrite['dia'].values, dims='points'), # interpolate to the exact McSnow properties
					                    aspect=xr.DataArray(mcTabledendrite['sPhi'].values, dims='points'),
					                    mass=xr.DataArray(mcTabledendrite['mTot'].values, dims='points'))
                    points['S22r_S11r'] = points.S22r - points.S11r 
                    reflect_h,  reflect_v, kdp_M1, ldr, rho_hv = radarScat(points, wl) # calculate scattering properties from Matrix entries

                    mcTable['sZeH'].loc[elv,wl,mcTabledendrite.index] = reflect_h
                    mcTable['sZeV'].loc[elv,wl,mcTabledendrite.index] = reflect_v
                    mcTable['sKDP'].loc[elv,wl,mcTabledendrite.index] = kdp_M1

                #- now for aggregates
                if len(mcTableAgg.mTot)>0: # only if aggregates are here
                    
                    elvSelAgg = scatSet['lutElevAgg'][np.argmin(np.abs(np.array(scatSet['lutElevAgg'])-elv))]
                    freSel = scatSet['lutFreqAgg'][np.argmin(np.abs(np.array(scatSet['lutFreqAgg'])-f/1e9))]# select correct frequency
                    freSel = str(freSel).ljust(6,'0')#
                    #print('frequency ', f/1.e9, 'lut frequency ', freSel)
                    dataset_filename = scatSet['lutPath'] + 'DDA_LUT_dendrite_aggregates_freq{}_elv{}.nc'.format(freSel,int(elvSelAgg))#, int(elvSelAgg)) 
                    lut = xr.open_dataset(dataset_filename)
                    lut = lut.sel(elevation = elv, wavelength=wl,method='nearest') # select closest elevation and wavelength
                    points = lut.interp(mass = xr.DataArray(mcTableAgg['mTot'].values, dims='points')) # interpolate to exact McSnow properties

                    points['S22r_S11r'] = points.S22r - points.S11r 
                    reflect_h,  reflect_v, kdp_M1, ldr, rho_hv = radarScat(points, wl) # get scattering properties from Matrix entries
                    #plt.plot(points.Dmax,reflect_h,marker='.',ls='None')
                    #plt.show()
                    mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = reflect_h
                    mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = reflect_v
                    mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = kdp_M1

    elif scatSet['mode'] == 'DDA_rational':
        #- this uses rational functions to calculate scattering properties. 
        #- requires: fitting parameters
        mcTabledendrite = mcTable[mcTable['sPhi']<1].copy()
        mcTableAgg = mcTable[(mcTable['sNmono']>1)].copy()
        for wl in wls:
            wlStr = '{:.2e}'.format(wl)
            if len(mcTabledendrite)>0:
                #elvSel = scatSet['lutElev'][np.argmin(np.abs(np.array(scatSet['lutElev'])-elv))] 
                fitting_params = '/project/meteo/work/L.Terzi/pol-scatt/DDA_dendrites/fitting_parameters_rationalFunc_p22_freq9.6000_elv30.txt'#.format(freSel, int(elvSel)) 
                pfitAll = pd.read_csv(fitting_params,delimiter=' ')
                # for all variables:
                pointsDic = {'Z11':0,'Z12':0,'Z21':0,'Z22':0,'S11r':0,'S22r':0,'S11i':0,'S22i':0}
                for var in pointsDic.keys():
                    pfit = pfitAll[var]
                    afit,bfit = get_ab_from_params(pfit.values)
                    fit = rational3d(np.log10(mcTabledendrite.mTot.values),np.log10(mcTabledendrite.dia.values),np.log10(mcTabledendrite.sPhi.values),afit,bfit)
                    if var in ['Z11','Z12','Z21','Z22']:
                        fit = 10**fit*1e6 # in mm**2
                    else:
                        fit = 10**fit*1e3  #in mm
                    pointsDic[var] = fit
                #print('elevation ', elv,'lut elevation ', elvSel)        
                points = pd.DataFrame.from_dict(pointsDic)
                #print(points)
                points['S22r_S11r'] = points.S22r - points.S11r 
                
                reflect_h,  reflect_v, kdp_M1, ldr, rho_hv = radarScat(points, wl)
                
                mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = reflect_h 
                mcTable['sZeV_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = reflect_v
                mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = kdp_M1
            else:
                mcTable['sZeH_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = np.nan
                mcTable['sZeV_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = np.nan
                mcTable['sKDP_{0}'.format(wlStr)].values[mcTable['sPhi']<1] = np.nan
    elif scatSet['mode'] == 'interpolate': # interpolation fails if no selection is possible
        elvSel = scatSet['lutElev'][np.argmin(np.abs(np.array(scatSet['lutElev'])-elv))]
        print('elevation ', elv,'lut elevation ', elvSel)
        
        for wl in wls:
            f = 299792458e3/wl
            freSel = scatSet['lutFreq'][np.argmin(np.abs(np.array(scatSet['lutFreq'])-f))]
            print('frequency ', f/1.e9, 'lut frequency ', freSel/1.e9)
            dataset_filename = scatSet['lutPath'] + 'testLUT_{:3.1f}e9Hz_{:d}.nc'.format(freSel/1e9, int(elvSel)) 
            
            # this is the wisdom part, only calculate if particle has not been filled yet!
            lut = xr.open_dataset(dataset_filename).load()#.sel(wavelength=wl,
                                                   #     elevation=elv,
                                                   #     canting=1.0,
                                                   #     method='nearest')

            #print(lut)
            
            points = lut.sel(wavelength=wl, elevation=elv, canting=1.0,
                         size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
                         aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
                         density=xr.DataArray(mcTable['sRho_tot_g'].values, dims='points'),
                         method='nearest')
            
            
            
            #tmpTable = 
            # for ar < 1!!
            canting = True
            meanAngle=0
            cantingStd=1
            points1 = points.where(points.aspect < 1,drop=True)
            size1 = xr.DataArray(mcTable['radii_mm'].values, dims='points').where(points.aspect < 1,drop=True).values
            ar1 = xr.DataArray(mcTable['sPhi'].values, dims='points').where(points.aspect < 1,drop=True).values
            rho1 = xr.DataArray(mcTable['sRho_tot_g'].values, dims='points').where(points.aspect < 1,drop=True).values
            if len(points1.size)>0:
                radii_M1 = points1.radii_mm.where(np.isnan(points1.Z11),drop=True).values #[mm]
                if len(radii_M1)>0:
                    print('updating LUT with ',len(radii_M1),' new particles')
                    as_ratio_M1 = points1.aspect.where(np.isnan(points1.Z11),drop=True).values
                    rho_M1 = points1.density.where(np.isnan(points1.Z11),drop=True).values
                    ar1 = ar1.where(np.isnan(points1.Z11),drop=True).values
                    rho1 = rho1.where(np.isnan(points1.Z11),drop=True).values
                    size1 = size1.where(np.isnan(points1.Z11),drop=True).values
                    singleScat = calcScatPropOneFreq(wl, size1, ar1, 
                                         rho1, elv, canting=canting, 
                                         cantingStd=cantingStd, 
                                         meanAngle=meanAngle, ndgs=ndgs,
                                         safeTmatrix=scatSet['safeTmatrix'])
                    reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
            
            # fill LUT with new values
                    for i in range(len(radii_M1)):
                        #print(i, 'of total ',len(radii_M1))
                        lut.Z11.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z11Mat[i]
                        lut.Z12.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z12Mat[i]
                        lut.Z21.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z21Mat[i]
                        lut.Z22.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z22Mat[i]
                        lut.Z33.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z33Mat[i]
                        lut.Z44.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z44Mat[i]
                        lut.S11i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S11iMat[i]
                        lut.S22i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S22iMat[i]
                        lut.S22r_S11r.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = sMat[i]
            
            # for ar >= 1!!
            canting = True
            meanAngle=90
            cantingStd=1
            points1 = points.where(points.aspect >= 1,drop=True)
            
            if len(points1.size)>0:
                radii_M1 = points1['size'].where(np.isnan(points1.Z11),drop=True).values #[mm]
                
                as_ratio_M1 = points1['aspect'].where(np.isnan(points1.Z11),drop=True).values
                rho_M1 = points1['density'].where(np.isnan(points1.Z11),drop=True).values
                if len(radii_M1)>0:
                    print('updating LUT with ',len(radii_M1),' new particles')
                    singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                         rho_M1, elv, canting=canting, 
                                         cantingStd=cantingStd, 
                                         meanAngle=meanAngle, ndgs=ndgs,
                                         safeTmatrix=scatSet['safeTmatrix'])
                    reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
            
                # fill LUT with new values
                    for i in range(len(radii_M1)):
                        #print(i, 'of total ',len(radii_M1)-1)
                        lut.Z11.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z11Mat[i]
                        lut.Z12.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z12Mat[i]
                        lut.Z21.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z21Mat[i]
                        lut.Z22.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z22Mat[i]
                        lut.Z33.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z33Mat[i]
                        lut.Z44.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z44Mat[i]
                        lut.S11i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S11iMat[i]
                        lut.S22i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S22iMat[i]
                        lut.S22r_S11r.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = sMat[i]
                    lut.close()
                    lut.to_netcdf(dataset_filename)
            
            
            lut = xr.open_dataset(dataset_filename)
            #print(lut)
            
            # now select new points in lut and calculate reflect,...
            points = lut.sel(wavelength=wl, elevation=elv, canting=1.0,
                             size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
                             aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
                             density=xr.DataArray(mcTable['sRho_tot_g'].values, dims='points'),
                             method='nearest')
            
            #- interpolate rather than select!!
            #lutWaveEl = lut.sel(wavelength=wl, elevation=elv, canting=1.0)
            
                
            #pointsnew = lutWaveEl.interp(size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
            #			             aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
           # 			             density=xr.DataArray(mcTable['sRho_tot_g'].values, dims='points'))
            
            
            reflect_h,  reflect_v, kdp_M1, ldr, rho_hv = radarScat(points, wl)
            
            wlStr = '{:.2e}'.format(wl)
            
            mcTable['sZeH_{0}'.format(wlStr)] = reflect_h
            mcTable['sZeV_{0}'.format(wlStr)] = reflect_v
            mcTable['sKDP_{0}'.format(wlStr)] = kdp_M1
    
    
    
    
    
    
    elif len(mcTable): # interpolation fails if no selection is possible
        elvSel = scatSet['lutElev'][np.argmin(np.abs(np.array(scatSet['lutElev'])-elv))]
        print('elevation ', elv,'lut elevation ', elvSel)
        #if scatSet['mode'] == 'table':
        #    print('fast LUT mode')

        #elif scatSet['mode'] == 'wisdom':            
        #    print('less fast cache adaptive mode')
        
        for wl in wls:
            f = 299792458e3/wl
            freSel = scatSet['lutFreq'][np.argmin(np.abs(np.array(scatSet['lutFreq'])-f))]
            print('frequency ', f/1.e9, 'lut frequency ', freSel/1.e9)
            dataset_filename = scatSet['lutPath'] + 'testLUT_{:3.1f}e9Hz_{:d}.nc'.format(freSel/1e9, int(elvSel)) 
            lut = xr.open_dataset(dataset_filename).load()#.sel(wavelength=wl,
                                                   #     elevation=elv,
                                                   #     canting=1.0,
                                                   #     method='nearest')

            #print(lut)
            if scatSet['mode'] == 'wisdom':
                points = lut.sel(wavelength=wl, elevation=elv, canting=1.0,
                             size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
                             aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
                             density=xr.DataArray(mcTable['sRho_tot_g'].values, dims='points'),
                             method='nearest')
                
                
                #tmpTable = 
                # for ar < 1!!
                canting = True
                meanAngle=0
                cantingStd=1
                points1 = points.where(points.aspect < 1,drop=True)
                if len(points1.size)>0:
                    radii_M1 = points1['size'].where(np.isnan(points1.Z11),drop=True).values #[mm]
                    if len(radii_M1)>0:
                        print('updating LUT with ',len(radii_M1),' new particles')
                        as_ratio_M1 = points1['aspect'].where(np.isnan(points1.Z11),drop=True).values
                        rho_M1 = points1['density'].where(np.isnan(points1.Z11),drop=True).values
                        singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                             rho_M1, elv, canting=canting, 
                                             cantingStd=cantingStd, 
                                             meanAngle=meanAngle, ndgs=ndgs,
                                             safeTmatrix=scatSet['safeTmatrix'])
                        reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
                
                # fill LUT with new values
                        for i in range(len(radii_M1)):
                            #print(i, 'of total ',len(radii_M1))
                            lut.Z11.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z11Mat[i]
                            lut.Z12.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z12Mat[i]
                            lut.Z21.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z21Mat[i]
                            lut.Z22.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z22Mat[i]
                            lut.Z33.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z33Mat[i]
                            lut.Z44.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z44Mat[i]
                            lut.S11i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S11iMat[i]
                            lut.S22i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S22iMat[i]
                            lut.S22r_S11r.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = sMat[i]
                
                # for ar >= 1!!
                canting = True
                meanAngle=90
                cantingStd=1
                points1 = points.where(points.aspect >= 1,drop=True)
                
                if len(points1.size)>0:
                    radii_M1 = points1['size'].where(np.isnan(points1.Z11),drop=True).values #[mm]
                    
                    as_ratio_M1 = points1['aspect'].where(np.isnan(points1.Z11),drop=True).values
                    rho_M1 = points1['density'].where(np.isnan(points1.Z11),drop=True).values
                    if len(radii_M1)>0:
                        print('updating LUT with ',len(radii_M1),' new particles')
                        singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                             rho_M1, elv, canting=canting, 
                                             cantingStd=cantingStd, 
                                             meanAngle=meanAngle, ndgs=ndgs,
                                             safeTmatrix=scatSet['safeTmatrix'])
                        reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
                
                    # fill LUT with new values
                        for i in range(len(radii_M1)):
                            #print(i, 'of total ',len(radii_M1)-1)
                            lut.Z11.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z11Mat[i]
                            lut.Z12.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z12Mat[i]
                            lut.Z21.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z21Mat[i]
                            lut.Z22.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z22Mat[i]
                            lut.Z33.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z33Mat[i]
                            lut.Z44.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = Z44Mat[i]
                            lut.S11i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S11iMat[i]
                            lut.S22i.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = S22iMat[i]
                            lut.S22r_S11r.loc[radii_M1[i], as_ratio_M1[i], rho_M1[i], float(wl),float(elv),float(canting)] = sMat[i]
                        lut.close()
                        lut.to_netcdf(dataset_filename)
            lut = xr.open_dataset(dataset_filename)
            #print(lut)
            
            # now select new points in lut and calculate reflect,...
            points = lut.sel(wavelength=wl, elevation=elv, canting=1.0,
                             size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
                             aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
                             density=xr.DataArray(mcTable['sRho_tot_g'].values, dims='points'),
                             method='nearest')
            
            #- interpolate rather than select!!
            #lutWaveEl = lut.sel(wavelength=wl, elevation=elv, canting=1.0)
            
                
            #pointsnew = lutWaveEl.interp(size=xr.DataArray(mcTable['radii_mm'].values, dims='points'),
            #			             aspect=xr.DataArray(mcTable['sPhi'].values, dims='points'),
           # 			             density=xr.DataArray(mcTable['sRho_tot_g'].values, dims='points'))
            
            
            reflect_h,  reflect_v, kdp_M1, ldr, rho_hv = radarScat(points, wl)
            
            wlStr = '{:.2e}'.format(wl)
            
            mcTable['sZeH_{0}'.format(wlStr)] = reflect_h
            mcTable['sZeV_{0}'.format(wlStr)] = reflect_v
            mcTable['sKDP_{0}'.format(wlStr)] = kdp_M1
            #quit()
            

    return mcTable


