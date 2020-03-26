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
import mcradar as mcr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.style.use('seaborn')

print('-------------------------------')
print('Starting the installation tests')
print('-------------------------------')


print('loading the settings')
dicSettings = mcr.loadSettings(dataPath='data_test.dat',
                               elv=30, freq=np.array([95e9]))


print('loading the McSnow output')
mcTable = mcr.getMcSnowTable(dicSettings['dataPath'])


print('selecting time step = 600 min  ')
selTime = 600.
times = mcTable['time']
mcTable = mcTable[times==selTime]
mcTable = mcTable.sort_values('sHeight')


print('getting things done :) -> testing the fullRadarOperator')
output = mcr.fullRadar(dicSettings, mcTable)

print('saving the output file')
output.to_netcdf('output.nc')

print('plotting the spetra')
for wl in dicSettings['wl']:

    wlStr = '{:.2e}'.format(wl)
    plt.figure(figsize=(8,8))
    mcr.lin2db(output['spec_H_{0}'.format(wlStr)]).plot(vmin=-50, vmax=0)
    plt.title('Ze_H McSnow rad: {0} elv: {1} at time step: {2}'.format(wlStr, dicSettings['elv'], selTime))
    plt.ylim(dicSettings['minHeight'], dicSettings['maxHeight'])
    plt.xlim(-3, 0)
    plt.grid(b=True)
    plt.savefig('test_Ze_H_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
    plt.close()


print('plotting the KDP')
for wl in dicSettings['wl']:

    wlStr = '{:.2e}'.format(wl)

    plt.figure(figsize=(7,8))
    output['kdpInt_{0}'.format(wlStr)].plot(y='range', lw=3)
    plt.title('KDP McSnow rad: {0} elv: {1} at time step: {2}'.format(wlStr, dicSettings['elv'], selTime))
    plt.ylim(dicSettings['minHeight'], dicSettings['maxHeight'])
    plt.grid(b=True)
    plt.savefig('test_KDP_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('----------------')
print('the test is done')
print('----------------')


