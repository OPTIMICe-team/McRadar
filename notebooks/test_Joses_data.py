# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: José Dias Neto


# This is a testing code
# If every thing went well during 
# the installation of the package 
# you will get a 2 final plots
# one for the spectra and another
# one for KDP.

import numpy as np
import xarray as xr
import mcradar as mcr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.style.use('seaborn')

def getApectRatio(radii):
    # imput radii [mm]
    
    # auer et all 1970 (The Dimension of Ice Crystals in Natural Clouds)
    diameter = 2 * radii *1e3 # calculating the diameter in [mu m]
    h = 2.020 * (diameter)**0.449

    as_ratio = h / diameter
    
    return as_ratio

print('loading the settings')
dataPath = '/net/broebroe/lvonterz/BIMOD/4Jose/mass2fr_0300-0600min_avtstep_5.ncdf'
dicSettings = mcr.loadSettings(dataPath=dataPath,
                               elv=30, freq=np.array([9.6e9]),gridBaseArea=5.0,heightRes=77)


print('loading the McSnow output')
data = data = xr.open_dataset(dataPath)
time = np.ones_like(data.dim_SP_all_av150)
sPhi = np.ones_like(data.dim_SP_all_av150)*np.nan
sPhi = getApectRatio(data.diam * 1e3)
sPhi[data.mm.values > 1]=0.6
sPhi[data.mm.values == 1]=0.1
dataTable = data.to_dataframe()
dataTable = dataTable.rename(columns={'m_tot':'mTot', 'height':'sHeight', 
                                      'vt':'vel', 'diam':'dia','xi':'sMult'})
#adding required variables
dataTable['time']=time
dataTable['radii_mm'] = dataTable['dia'] * 1e3 /2.# particle radius in mm 
dataTable['mTot_g'] = dataTable['mTot'] * 1e3 # mass in grams
dataTable['dia_cm'] = dataTable['dia'] * 1e2 # diameter in centimeters
dataTable['sPhi']=sPhi
dataTable = dataTable[(dataTable['sPhi'] >= 0.015)]
#calculating density
dataTable = mcr.tableOperator.calcRho(dataTable)

print('getting things done :) -> testing the fullRadarOperator')
output = mcr.fullRadar(dicSettings, dataTable)

for wl in dicSettings['wl']:
    wlStr = '{:.2e}'.format(wl)
    output['Ze_H_{0}'.format(wlStr)] = output['spec_H_{0}'.format(wlStr)].sum(dim='vel')
    output['Ze_V_{0}'.format(wlStr)] = output['spec_V_{0}'.format(wlStr)].sum(dim='vel')
print(output)

print('saving the output file')
output.to_netcdf('output/output_Jose_test.nc')

print('plotting the spetra')
for wl in dicSettings['wl']:

    wlStr = '{:.2e}'.format(wl)
    plt.figure(figsize=(8,8))
    mcr.lin2db(output['spec_H_{0}'.format(wlStr)]).plot(vmin=-30, vmax=10)
    plt.title('Ze_H_spec McSnow rad: {0} elv: {1} at time step: {2}'.format(wlStr, dicSettings['elv'], selTime))
    plt.ylim(dicSettings['minHeight'], dicSettings['maxHeight'])
    plt.xlim(-3, 0)
    plt.grid(b=True)
    plt.savefig('plots/Jose_test_spec_Ze_H_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
    plt.close()
print('plotting the ZDR spetra')
for wl in dicSettings['wl']:

    wlStr = '{:.2e}'.format(wl)
    plt.figure(figsize=(8,8))
    (mcr.lin2db(output['spec_H_{0}'.format(wlStr)])-mcr.lin2db(output['spec_V_{0}'.format(wlStr)])).plot(vmin=-3, vmax=1)
    plt.title('ZDR McSnow rad: {0} elv: {1} at time step: {2}'.format(wlStr, dicSettings['elv'], selTime))
    plt.ylim(dicSettings['minHeight'], dicSettings['maxHeight'])
    plt.xlim(-3, 0)
    plt.grid(b=True)
    plt.savefig('plots/Jose_test_spec_ZDR_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('plotting Ze')
for wl in dicSettings['wl']:

    wlStr = '{:.2e}'.format(wl)
    plt.figure(figsize=(8,8))
    plt.plot(mcr.lin2db(output['Ze_H_{0}'.format(wlStr)]),output['range'])
    plt.xlabel('Z_H [dB]')
    plt.ylabel('range [m]')
    plt.title('Ze_H McSnow rad: {0} elv: {1} at time step: {2}'.format(wlStr, dicSettings['elv'], selTime))
    plt.ylim(dicSettings['minHeight'], dicSettings['maxHeight'])
    #plt.xlim(-3, 0)
    plt.grid(b=True)
    plt.savefig('plots/Jose_test_Ze_H_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
    plt.close()


print('plotting the KDP')
for wl in dicSettings['wl']:

    wlStr = '{:.2e}'.format(wl)

    plt.figure(figsize=(7,8))
    output['kdpInt_{0}'.format(wlStr)].plot(y='range', lw=3)
    plt.title('KDP McSnow rad: {0} elv: {1} at time step: {2}'.format(wlStr, dicSettings['elv'], selTime))
    plt.ylim(dicSettings['minHeight'], dicSettings['maxHeight'])
    plt.grid(b=True)
    plt.savefig('plots/Jose_test_KDP_{0}_no_vol.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('----------------')
print('the test is done')
print('----------------')



