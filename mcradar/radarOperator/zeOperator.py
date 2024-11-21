# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess
import numpy as np
import xarray as xr
from glob import glob
from scipy import constants
from mcradar.tableOperator import creatRadarCols
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors

debugging = False
onlyInterp = False

# TODO: this function should deal with the LUTs
def calcScatPropOneFreq(wl, radii, as_ratio, 
                        rho, elv, ndgs=30,
                        canting=False, cantingStd=1, 
                        meanAngle=0, safeTmatrix=False):
    from pytmatrix.tmatrix import Scatterer
    from pytmatrix import psd, orientation, radar
    from pytmatrix import refractive, tmatrix_aux

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
    for i, radius in enumerate(radii): 
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


def radarScat(sp, wl, K2):
#TODO check if K2 is for ice or liquid!
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
    rho_hv: correlation coefficient (array[n])
    """
    prefactor = 2*np.pi*wl**4/(np.pi**5*K2)
    
    
    reflect_hh = prefactor*(sp['Z11']+sp['Z22']+sp['Z12']+sp['Z21'])
    reflect_vv = prefactor*(sp['Z11']+sp['Z22']-sp['Z12']-sp['Z21'])
    kdp = 1e-3*(180.0/np.pi)*wl*(sp['S22r'] - sp['S11r'])

    reflect_hv = prefactor*(sp['Z11'] - sp['Z12'] + sp['Z21'] - sp['Z22'])
    #reflect_vh = prefactor*(sp.Z11 + sp.Z12 - sp.Z21 - sp.Z22).values
               
    # delta_hv np.arctan2(Z[2,3] - Z[3,2], -Z[2,2] - Z[3,3])
    #a = (Z[2,2] + Z[3,3])**2 + (Z[3,2] - Z[2,3])**2
    #b = (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    #c = (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])
    #rho_hv np.sqrt(a / (b*c))
    rho_hv = np.nan*np.ones_like(reflect_hh) # disable rho_hv for now
    #Ah = 4.343e-3 * 2 * scatterer.wavelength * sp.S22i.values # attenuation horizontal polarization
    #Av = 4.343e-3 * 2 * scatterer.wavelength * sp.S11i.values # attenuation vertical polarization

    #- test: calculate extinction: TODO: test Cextx that is given in DDA with this calculation.
    k = 2 * np.pi / (wl)
    cext_hh = sp['S22i']*4.0*np.pi/k
    cext_vv = sp['S11i']*4.0*np.pi/k
    
    return reflect_hh, reflect_vv, reflect_hv, kdp, rho_hv, cext_hh, cext_vv


def calcParticleZe(wls, elvs, mcTable,scatSet,beta,beta_std):#zeOperator
    """
    Calculates the horizontal and vertical reflectivity of 
    each superparticle from a given distribution of super 
    particles,in this case I just quickly wanted to change the function to deal with Monomers with the DDA LUT and use Tmatrix for the aggregates
    
    Parameters
    ----------
    wls: wavelength [mm] (iterable)
    elv: elevation angle [°] # TODO: maybe also this can become iterable
    mcTable: McSnow table returned from getMcSnowTable()
    scatSet: type of scattering calculations to use, choose between full and DDA
    orientational_avg: boolean to choose if the scattering properties are averaged over multiple orientations
    beta: mean canting angle of particle
    beta_std= standard deviation of canting angle of particle
    Returns 
    -------
    mcTable including the horizontal and vertical reflectivity
    of each super particle calculated for X, Ka and W band. The
    calculation is made separetely for aspect ratio < 1 and >=1.
    Kdp is also included. TODO spectral ldr and rho_hv
    """
    
    #calling the function to create output columns

    if scatSet['mode'] == 'Tmatrix':
        print('Full mode Tmatrix calculation')
        ##calculation of the reflectivity for AR < 1
        
        tmpTable = mcTable.where(mcTable['sPhi']<1,drop=True)
        #particle properties
        canting = True
        meanAngle=0
        cantingStd=1
        radii_M1 = tmpTable['radii_mm'].values #[mm]
        as_ratio_M1 = tmpTable['sPhi'].values
        rho_M1 = tmpTable['sRho_tot_gcm'].values #[g/mm^3]
        for wl in wls:
            prefactor = 2*np.pi*wl**4/(np.pi**5*0.93)
        
            for elv in elvs:            
                singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                                 rho_M1, elv, canting=canting, 
                                                 cantingStd=cantingStd, 
                                                 meanAngle=meanAngle, ndgs=scatSet['ndgs'],
                                                 safeTmatrix=scatSet['safeTmatrix'])
                reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
                reflect_hv = prefactor*(Z11Mat - Z12Mat + Z21Mat - Z22Mat)
                wlStr = '{:.2e}'.format(wl)
                #print('reflect done')
                #print(tmpTable.index)
                #print(tmpTable)
                mcTable['sZeH'].loc[elv,wl,tmpTable.index] = reflect_h
                mcTable['sZeV'].loc[elv,wl,tmpTable.index] = reflect_v
                mcTable['sZeHV'].loc[elv,wl,tmpTable.index] = reflect_hv
                mcTable['sKDP'].loc[elv,wl,tmpTable.index] = kdp_M1

        
        ##calculation of the reflectivity for AR >= 1
        tmpTable = mcTable.where(mcTable['sPhi']>=1,drop=True)
        canting=True
        meanAngle=90
        cantingStd=1
        radii_M1 = (tmpTable['radii_mm']).values #[mm]
        as_ratio_M1 = tmpTable['sPhi'].values
        rho_M1 = tmpTable['sRho_tot_gcm'].values #[g/mm^3]
        for wl in wls:
            prefactor = 2*np.pi*wl**4/(np.pi**5*0.93)
            for elv in elvs:     
                singleScat = calcScatPropOneFreq(wl, radii_M1, as_ratio_M1, 
                                                 rho_M1, elv, canting=canting, 
                                                 cantingStd=cantingStd, 
                                                 meanAngle=meanAngle, ndgs=scatSet['ndgs'],
                                                 safeTmatrix=scatSet['safeTmatrix'])
                reflect_h, reflect_v, refInd, kdp_M1, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = singleScat
                reflect_hv = prefactor*(Z11Mat - Z12Mat + Z21Mat - Z22Mat)
                print('reflect done')
            
                wlStr = '{:.2e}'.format(wl)
                mcTable['sZeH'].loc[elv,wl,tmpTable.index] = reflect_h
                mcTable['sZeV'].loc[elv,wl,tmpTable.index] = reflect_v
                mcTable['sZeHV'].loc[elv,wl,tmpTable.index] = reflect_hv
                mcTable['sKDP'].loc[elv,wl,tmpTable.index] = kdp_M1
    
    elif scatSet['mode'] == 'DDA':
        """
        #-- this option uses the output of the DDA calculations. 
        We are reading in all data, then selecting the corresponding wl, elevation.
        Then, you can choose how you want your points selected out of the table. 
        We have the option to select the n closest neighbours and average over them, 
        to define a radius in which all values are taken and averaged,
        or you can choose a nearest neighbour regression which chooses n closest neighbours and wheights the average with the inverse distance of the points. 
        """
        scatPoints={}
        # different DDA LUT for monomers and Aggregates. 
        mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True) # select only cry
        #print(mcTableCry)
        #print(mcTableCry.dia,mcTableCry.mTot,mcTableCry.sPhi)
        #mcTablePlate = mcTableCry.where(mcTableCry['sPhi']<=1,drop=True) # select only plates
        #mcTableColumn = mcTableCry.where(mcTableCry['sPhi']>1,drop=True) # select only needle 
        mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True) # select only aggregates
        rimed = False # TODO: make that dependent on rime mass fraction!
        betas = np.random.normal(loc=beta, scale=beta_std, size=len(mcTableCry.dia))

        if scatSet['orientational_avg'] == False:
            DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals.nc')
            DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'scattering_properties_plate_aggs.nc')
        else:
            DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals_withbeta.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
            DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'scattering_properties_plate_aggs.nc')
        
        DDA_data_cry = DDA_data_cry.to_dataframe()
        DDA_data_agg = DDA_data_agg.to_dataframe()
        #print(DDA_data)
        # generate points to look up in the DDA LUT
        #fig,ax = plt.subplots(ncols=2,figsize=(10,5),constrained_layout=True)
        for wl in wls:
            wl_close = DDA_data_agg.iloc[(DDA_data_agg['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_agg = DDA_data_agg[DDA_data_agg.wavelength==wl_close]

            wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
            
            for elv in elvs:
                #print(DDA_wl)
                el_close = DDA_wl_agg.iloc[(DDA_wl_agg['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
                DDA_elv_agg = DDA_wl_agg[DDA_wl_agg.elevation==el_close]

                el_close = DDA_wl_cry.iloc[(DDA_wl_cry['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
                DDA_elv_cry = DDA_wl_cry[DDA_wl_cry.elevation==el_close]
                
                if scatSet['orientational_avg'] == False:

                    if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                        pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.D_max), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar))))
                        mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi))))
                        # select now the points according to the defined method
                        # Fit the KNeighborsRegressor
                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            scatPoints = {'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                            'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                            'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                            'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                            
                                            'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                            'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                            'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                            'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22r.values))-1}
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsCry)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1}

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.kneighbors(mcSnowPointsCry)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1}
                                        
                        # calculate scattering properties from Matrix entries                
                        reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                        mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = reflect_h#points.ZeH
                        mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = reflect_v#points.ZeV
                        mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = reflect_hv
                        mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = kdp_M1#points.KDP
                        mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = cext_hh
                        mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = cext_vv

                        #if elv == 30 and wl == wls[2]:
                        #    plt.plot(mcTableCry.dia,10*np.log10(reflect_h/reflect_v),marker='.',ls='None',label='cry, zdr, '+str(elv))
                        #    plt.legend()
                        #    plt.show()
                        #    ax[3].plot(mcTableCry.dia,scatPoints['kdp'],marker='.',ls='None',label='aggs, kdp, '+str(elv))
                        #    ax[3].legend()
                    #- now for aggregates
                    if len(mcTableAgg.mTot)>0: # only if aggregates are here
                        # only used rimed particles if riming is True. TODO: make that dependent on riming fraction
                        if rimed:
                            DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==1]
                        else:
                            DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==0]
                        
                        pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.D_max), np.log10(DDA_elv_agg.mass)))) # we need to differentiate here because for aggregates we are only selecting with mass and Dmax
                        mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot)))) 

                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            #print(points)
                            #print(np.log10(DDA_elv.Z11.values))
                            #print(mcSnowPoints)
                            #quit()
                            scatPoints = {'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                                            'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                            'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                            'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                            
                                            'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                            'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                            'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                            'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22r.values))-1}
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsAgg)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1}

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.kneighbors(mcSnowPointsAgg)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1}
                                        
                        
                        reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # get scattering properties from Matrix entries
                        #if elv == 90:
                        #   plt.plot(mcTableAgg.dia,10*np.log10(reflect_h),marker='.',ls='None',label='wl: '+str(wl)+' elv: '+str(elv))
                        mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = reflect_h
                        mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = cext_hh
                        mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = cext_vv
                        mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = reflect_v
                        mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = reflect_hv
                        mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = kdp_M1
                    
                else:# if we do orientational averaging:
                
                
                    #print(betas)
                    #quit()
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar),DDA_elv_cry.beta)))
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi),betas)))
                    #scatPoints={}
                    
                    if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                        # select now the points according to the defined method
                        # Fit the KNeighborsRegressor
                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            scatPoints = {'cbck_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.c_bck_h.values)).predict(mcSnowPointsCry)),#'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                        # 'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                            #'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                            #'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                            #'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                            #'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                            #'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                            #'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                            
                                            'cbck_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.c_bck_v.values)).predict(mcSnowPointsCry)),
                                            'cbck_hv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.c_bck_hv.values+abs(np.min(DDA_elv_cry.c_bck_hv.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.c_bck_hv.values))-1,
                                            'cext_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_hh.values+abs(np.min(DDA_elv_cry.cext_hh.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.cext_hh.values))-1,
                                            'cext_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_vv.values+abs(np.min(DDA_elv_cry.cext_vv.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.cext_vv.values))-1,
                                            'kdp':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.kdp.values+abs(np.min(DDA_elv_cry.kdp.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.kdp.values))-1,
                                            }
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsCry)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_cry.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_cry.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_cry.c_bck_hv.values+abs(np.min(DDA_elv_cry.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_cry.cext_hh.values+abs(np.min(DDA_elv_cry.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_cry.cext_vv.values+abs(np.min(DDA_elv_cry.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_cry.kdp.values+abs(np.min(DDA_elv_cry.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.kdp.values))-1}
                            

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.kneighbors(mcSnowPointsCry)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_cry.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_cry.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_cry.c_bck_hv.values+abs(np.min(DDA_elv_cry.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_cry.cext_hh.values+abs(np.min(DDA_elv_cry.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_cry.cext_vv.values+abs(np.min(DDA_elv_cry.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_cry.kdp.values+abs(np.min(DDA_elv_cry.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.kdp.values))-1}
                                        
                        # calculate scattering properties from Matrix entries                
                        #reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                        
                        #mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = reflect_h#points.ZeH
                        mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_h']
                        mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_h']
                        mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_v']
                        mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_v']
                        mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_hv']
                        mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = scatPoints['kdp']
                        
                        #if elv == 90:# and wl==wls[2]:
                            #ax[0].plot(mcTableCry.dia,10*np.log10(scatPoints['cbck_h']),marker='.',ls='None',label='cry, h, '+str(wl))
                        #    ax[0].plot(mcTableCry.dia,scatPoints.kdp,marker='.',ls='None',label='cry, kdp, '+str(wl))
                        #    ax[0].legend()
                            #plt.plot(mcTableCry.dia,10*np.log10(scatPoints['cbck_v']),marker='.',ls='None',label='cry, v, '+str(wl))
                        #if elv == 30 and wl == wls[2]:
                        #    ax[2].plot(mcTableCry.dia,10*np.log10(scatPoints['cbck_h']/scatPoints['cbck_v']),marker='.',ls='None',label='cry, zdr, '+str(elv))
                        #    ax[2].legend()
                        #    ax[3].plot(mcTableCry.dia,scatPoints['kdp'],marker='.',ls='None',label='aggs, kdp, '+str(elv))
                        #    ax[3].legend()
                    
                    #- now for aggregates
                    if len(mcTableAgg.mTot)>0: # only if aggregates are here
                        # only used rimed particles if riming is True. TODO: make that dependent on riming fraction
                        if rimed:
                                DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==1]
                        else:
                            DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==0]
                        scatPoints={}
                        pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.D_max), np.log10(DDA_elv_agg.mass)))) # we need to differentiate here because for aggregates we are only selecting with mass and Dmax
                        mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot)))) 
                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            #print(points)
                            #print(np.log10(DDA_elv.Z11.values))
                            #print(mcSnowPoints)
                            #quit()
                            scatPoints = {'cbck_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_h.values)).predict(mcSnowPointsAgg)),#'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                                        #  'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        #  'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        #  'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        #  'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        #  'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        #  'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        #  'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                            
                                            'cbck_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_v.values)).predict(mcSnowPointsAgg)),
                                            'cbck_hv':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                            'cext_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                            'cext_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                            'kdp':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.kdp.values))-1}
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsAgg)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_agg.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_agg.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.kdp.values))-1}

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.kneighbors(mcSnowPointsAgg)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_agg.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_agg.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.kdp.values))-1}
                                        
                        
                        #reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # get scattering properties from Matrix entries
                        #if elv == 90:
                        #    ax[1].plot(mcTableAgg.dia,10*np.log10(scatPoints['cbck_h']),marker='.',ls='None',label='aggs, h, '+str(wl))
                        #    ax[1].legend()
                            #ax[0].plot(mcTableAgg.dia,10*np.log10(scatPoints['cbck_v']),marker='.',ls='None',label='aggs, v, '+str(wl))
                        #if elv == 30 and wl == wls[2]:
                        #    ax[2].plot(mcTableAgg.dia,10*np.log10(scatPoints['cbck_h']/scatPoints['cbck_v']),marker='.',ls='None',label='aggs, zdr, '+str(elv))
                        #    ax[2].legend()
                        #    ax[3].plot(mcTableAgg.dia,scatPoints['kdp'],marker='.',ls='None',label='aggs, kdp, '+str(elv))
                        #    ax[3].legend()
                        mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_h']
                        mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_h']
                        mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_v']
                        mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_v']
                        mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_hv']
                        mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = scatPoints['kdp']
                    
                    #plt.semilogx(mcTableAgg.dia,10*np.log10(mcTable.sZeH.loc[elv,wl,mcTableAgg.index]),marker='.',ls='None')
        #plt.legend()
        #plt.show()                
        #print('all calculations for all elv and wl took ', time.time() - t00,' seconds')
        #quit()
                
    
    
    
   
            

    return mcTable


