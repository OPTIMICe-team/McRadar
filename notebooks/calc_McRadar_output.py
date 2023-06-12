#-*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: Leonie von Terzi


# this calculates the polarimetric variables at Wband for McSnow output. 
# It is intended to test habit prediction, aggregation has not been implemented in this McSnow run.
# The McSnow data was produced by Jan-Niklas Welß

import numpy as np
import mcradar as mcr
from scipy import constants
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd
import xarray as xr

#freq = np.array([95e9])
freq = np.array([35.5e9])
experimentID = '1d___binary_xi1000_nz200_lwc0_iwc7_ncl10_nclmass10_ssat30_dtc5_nrp5_rm10_rt0_vt3_at0_stick2_dt1_h0-0_ba500'
inputPath = '/work/lvonterz/McSnow_habit/experiments/'+experimentID+'/'
print('loading the settings')

#-- load the settings of McSnow domain, as well as elevation you want to plot: 
#In order to avoid volume sampling problems, you have to insert the gridBaseArea as it was defined in the McSnow simulation
dicSettings = mcr.loadSettings(dataPath=inputPath+'mass2fr.dat',
                               #'../data/1d_habit_test_leonie.dat',#'../data/habit_1d_leonie.dat',#'../data/1d_habit_test_leonie.dat',#'../tests/data_test.dat',
                               elv=30, freq=freq,gridBaseArea=5.0,maxHeight=3850,ndgsVal=50,heightRes=50,scatSet={'mode':'full', 'safeTmatrix':False})

print('loading the McSnow output')
# now generate a table from the McSnow output. You can specify xi0, if it is not stored in the table (like in my cases)
mcTable = mcr.getMcSnowTable(dicSettings['dataPath'])
print('selecting time step = 600 min  ')
#-- now select time step to use (600s is usually used)
selTime = 600.
times = mcTable['time']
mcTable = mcTable[times==selTime]
mcTable = mcTable.sort_values('sHeight')
#print(mcTable.sPhi)
print('getting things done :) -> calculating radar variables')
#-- now the full polarimetric output is generated (KDP, aswell as spec_H and spec_V, from which you can calculate ZeH, ZeV, ZDR, sZDR)
#mcr.fullRadar(dicSettings,mcTable)
#quit()
output = mcr.fullRadar(dicSettings, mcTable)
print(output)
for wl in dicSettings['wl']:
    wlStr = '{:.2e}'.format(wl)
    output['Ze_H_{0}'.format(wlStr)] = output['spec_H_{0}'.format(wlStr)].sum(dim='vel')
    output['Ze_V_{0}'.format(wlStr)] = output['spec_V_{0}'.format(wlStr)].sum(dim='vel')
print(output)

print('saving the output file')
#-- now save it
output.to_netcdf(inputPath+'KaBand_output.nc')

