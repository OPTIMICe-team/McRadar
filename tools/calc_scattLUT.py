#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:11:42 2020

@author: dori
"""
import os
import socket
from datetime import datetime
import time

import pandas as pd
import xarray as xr
import numpy as np
import dask

from pytmatrix import tmatrix, refractive, orientation
start = time.time()

# Describe the dimensions of the thing
# First set of wavelength, to be saved on different LUTs
freq = np.array([1.8e9, 5.6e9, 9.6e9, 13.6e9, 35.6e9, 94.0e9, 220.0e9])
#freq = np.array([9.6e9, 35.6e9, 94.0e9])
wls = 299792458e3/freq
wls_label = ['S-band', 'X-band', 'Ku-band', 'Ka-band', 'W-band', 'G-band']
#wls_label = ['X-band', 'Ka-band', 'W-band']
wls_dict = dict(zip(wls_label, wls))

# Define elevation angles to be saved on different LUTs
elv = np.arange(0.0, 91.0, 5.0)

# Fixed coordinates for each loop
canting_std = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 40.0, 60.0])
canting = xr.IndexVariable(dims='canting', data=dask.array.from_array(canting_std),
                           attrs={'long_name':'standard deviation of a gaussiang distribution of wobbling angles',
                                  'units':'degrees'})
Dmax = np.arange(0.1, 30.001, 0.05)    
sizes = xr.IndexVariable(dims='size', data=dask.array.from_array(Dmax),
                         attrs={'long_name':'radius along the maximum dimension, half of maximum Diameter',
                                'units':'millimeters'})
Ar = np.arange(1, 20.01, 0.5)
Ar = np.concatenate([1./Ar[::-1], Ar[1:]])
aspects = xr.IndexVariable(dims='aspect', data=dask.array.from_array(Ar),
                           attrs={'long_name':'aspect ratio of the spheroid c/a, Ar>1 prolate Dmax==c. Ar<1 oblate Dmax==a',
                                  'units':'dimensionless'})
rho = np.arange(0.01, 0.921, 0.01)
densities = xr.IndexVariable(dims='density', data=dask.array.from_array(rho),
                             attrs={'long_name':'mass density of the spheroid',
                                    'units':'grams/centimeter**3'})

for fi, f in enumerate(freq):
    print('{:3.1f}e9 Hertz'.format(f/1e9), 'Elapsed time {} seconds'.format(int(time.time()-start)))
    wavelengths = xr.IndexVariable(dims='wavelength',
                                   data=dask.array.from_array([wls[fi]]),
                                   attrs={'long_name':'wavelength',
                                          'units':'millimeters'})
    for ei, e in enumerate(elv):
        print('elevation {:d} degrees'.format(int(e)), 'Elapsed time {} seconds'.format(int(time.time()-start)))
        elevation = xr.IndexVariable(dims='elevation',
                                     data=dask.array.from_array([e]),
                                     attrs={'long_name':'radar elevation angle measured from the horizon (0) to the zenith (90)',
                                            'units':'degrees'})
        dims = ['size', 'aspect', 'density', 'wavelength', 'elevation', 'canting']
        coords = dict(zip(dims, [sizes, aspects, densities, wavelengths, elevation, canting]))

        chunks = {'wavelength':1, 'elevation':1, 'canting':1}
        chunks = {'canting':1}  # with separate files wl and elv are not important

        empty_dask = dask.array.empty([coords[i].size for i in coords])
        # Backward scattering
        Z11 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 11 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z11 = Z11.chunk(chunks=chunks)
        Z12 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 12 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z12 = Z12.chunk(chunks=chunks)
        # this might be not necessary for particle symmetries
        Z21 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 21 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z21 = Z21.chunk(chunks=chunks)
        Z22 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 22 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z22 = Z22.chunk(chunks=chunks)
        Z33 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 33 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z33 = Z33.chunk(chunks=chunks)
        Z34 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 34 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z34 = Z34.chunk(chunks=chunks)
        # this might be not necessary for particle symmetries
        Z43 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 43 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z43 = Z43.chunk(chunks=chunks)
        Z44 = xr.DataArray(dims=dims, coords=coords,
                           data=dask.array.zeros_like(empty_dask)*np.nan,
                           attrs={'long_name':'element 44 of the scattering matrix at backward direction',
                                  'units':'millimeters**2'})
        Z44=Z44.chunk(chunks=chunks)
        # Forward scattering
        S22r_S11r = xr.DataArray(dims=dims, coords=coords,
                                 data=dask.array.zeros_like(empty_dask)*np.nan,
                                 attrs={'long_name':'difference of the real parts of elements 22 and 11 of the amplitude matrix at forward direction',
                                        'units':'millimeters'})
        S22r_S11r = S22r_S11r.chunk(chunks=chunks)
        S11i = xr.DataArray(dims=dims, coords=coords,
                            data=dask.array.zeros_like(empty_dask)*np.nan,
                            attrs={'long_name':'imaginary part of element 11 of the amplitude matrix at forward direction',
                                   'units':'millimeters'})
        S11i = S11i.chunk(chunks=chunks)
        S22i = xr.DataArray(dims=dims, coords=coords,
                            data=dask.array.zeros_like(empty_dask)*np.nan,
                            attrs={'long_name':'imaginary part of element 22 of the amplitude matrix at forward direction',
                                   'units':'millimeters'})
        S22i = S22i.chunk(chunks=chunks)
        
        ## COMPUTATIONS over the default dataset
        column_names = ['wl', # [31.557101, 8.565499, 3.155710]
                        'elv', # always 90
                        'meanAngle', # it is used only for prolate
                        'cantingStd', # always 1.0
                        'radius', # 8660
                        'rho', # 8706
                        'as_ratio' # 8769
                        ]
        
        data = pd.read_csv('/home/dori/table_McRadar.txt',
                           delim_whitespace=True,
                           header=None,
                           names=column_names) # 26463 lines
        data = data[np.abs((data.wl-wls[fi])/wls[fi])<0.05]
        data = data[data.elv == e]
        scatterer = tmatrix.Scatterer(wavelength=1.0)
        scatterer.radius_type = tmatrix.Scatterer.RADIUS_MAXIMUM
        scatterer.ndgs = 30
        scatterer.ddelta = 1e-6
        i = 0
        if len(data):
            for index, row in data.iterrows():
                coords = Z11.sel(wavelength=row.wl,
                                 size=row.radius,
                                 aspect=row.as_ratio,
                                 density=row.rho,
                                 elevation=row.elv,
                                 canting=row.cantingStd,
                                 method='nearest').coords
                if np.isnan(Z11.loc[coords]):
                    print(index, i)
                    i = i +1
                    scatterer.wavelength = row.wl
                    scatterer.radius = row.radius
                    scatterer.axis_ratio = 1./row.as_ratio
                    scatterer.m = refractive.mi(row.wl, row.rho)
                    scatterer.or_pdf = orientation.gaussian_pdf(std=row.cantingStd,
                                                                mean=int(row.as_ratio>1)*90.0)  
                    # scatterer.orient = orientation.orient_averaged_adaptive
                    scatterer.orient = orientation.orient_averaged_fixed
                    scatterer.thet0 = 90. - row.elv
                    scatterer.phi0 = 0.
                    
                    # First, backward
                    scatterer.thet = 180.0 - scatterer.thet0
                    scatterer.phi = (180. + scatterer.phi0) % 360.
                    Zmat = scatterer.get_Z()
                    Z11.load().loc[coords] = Zmat[0, 0]
                    Z22.load().loc[coords] = Zmat[1, 1]
                    Z21.load().loc[coords] = Zmat[1, 0]
                    Z12.load().loc[coords] = Zmat[0, 1]
                    Z33.load().loc[coords] = Zmat[2, 2]
                    Z34.load().loc[coords] = Zmat[2, 3]
                    Z43.load().loc[coords] = Zmat[3, 2]
                    Z44.load().loc[coords] = Zmat[3, 3]
                    
                    # Than, forward
                    scatterer.thet = scatterer.thet0
                    scatterer.phi = scatterer.phi0
                    Smat = scatterer.get_S()
                    S11i.load().loc[coords] = Smat[0, 0].imag
                    S22i.load().loc[coords] = Smat[1, 1].imag
                    S22r_S11r.load().loc[coords] = Smat[1, 1].real-Smat[0, 0].real
    
        ## Pack it up 
        variables = {'Z11':Z11,
                     'Z12':Z12,
                     'Z21':Z21,
                     'Z22':Z22,
                     'Z33':Z33,
                     'Z34':Z34,
                     'Z43':Z43,
                     'Z44':Z44,
                     'S22r_S11r':S22r_S11r,
                     'S11i':S11i,
                     'S22i':S22i}
        
        # maybe here I can calculate some optimal encoding???

        global_attributes = {'created_by':os.environ['USER'],
                             'host_machine':socket.gethostname(),
                             'software':'pytmatrix',
                             'refractive index model':'pytmatrix.refractive.mi',
                             #'frequency':'{:3.1f}e9'.format(w/1e9),
                             #'elevation_angle_degrees':elv,
                             #'standard_deviation_canting':canting_std,
                             'created_on':str(datetime.now()),
                             'comment':'this is the scattering Look Up Table to be used in McRadar to speed up the computation of the polarimetric scattering properties of snow'}

        dataset = xr.Dataset(data_vars=variables,
                             coords=coords,
                             attrs=global_attributes)
        filename = 'testLUT_{:3.1f}e9Hz_{:d}.nc'.format(f/1e9, int(e))
        encoding = dict([(v, {'complevel':5, 'zlib':True}) for v in variables.keys()])
        dataset.to_netcdf(filename, encoding=encoding)
#elev, elev_grp = zip(*dataset.groupby('elevation'))
#for e, eg in zip(elev, elev_grp):
#    wave, wave_grp = zip(*eg.groupby('wavelength'))
#paths = ["%s.nc" % y for y in years]
#xr.save_mfdataset(datasets, paths)
print('Execution time {} seconds'.format(int(time.time()-start)))

#%%
#lut = xr.open_dataset('/home/dori/develop/McRadar/tools/testLUT_{:3.1f}e9Hz_{:d}.nc'.format(9.6e9/1e9, int(90.0)))
#density = xr.DataArray([0.31, 0.31, 0.31], dims='points')
#size = xr.DataArray([0.21, 0.21, 0.21], dims='points')
#aspect = xr.DataArray([1.27, 1.24, 1.27], dims='points')
#aa = lut.sel(wavelength=31.23, elevation=90.0, canting=1.0, size=size, aspect=aspect, density=density, method='nearest')
#from pytmatrix import tmatrix, radar, refractive
#for a in range(aa.dims['points']):
#    print(a, aa.size.values[a])
#    scat = tmatrix.Scatterer()