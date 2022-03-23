# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: José Dias Neto


import xarray as xr
from mcradar import *


def fullRadar(dicSettings, mcTable):
    """
    Calculates the radar variables over the entire range

    Parameters
    ----------
    dicSettings: a dictionary with all settings output from loadSettings()
    mcTable: McSnow data output from getMcSnowTable()

    Returns
    -------
    specXR: xarray dataset with the spectra(range, vel) and KDP(range)
    """


    specXR = xr.Dataset()
    #specXR_turb = xr.Dataset()
    counts = np.ones_like(dicSettings['heightRange'])*np.nan
    vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']

    for i, heightEdge0 in enumerate(dicSettings['heightRange']):

        heightEdge1 = heightEdge0 + dicSettings['heightRes']
        
        print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
        mcTableTmp = mcTable[(mcTable['sHeight']>heightEdge0) &
                             (mcTable['sHeight']<=heightEdge1)].copy()
        #print('max sPhi',max(mcTableTmp.sPhi))
        #print('min sPhi',min(mcTableTmp.sPhi))

        #mcTableTmp = mcTableTmp[(mcTableTmp['sPhi']<=4)]
        mcTableTmp = mcTableTmp[(mcTableTmp['sPhi']>=0.01)]
        
        mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
                                    mcTableTmp, ndgs=dicSettings['ndgsVal'],
                                    scatSet=dicSettings['scatSet'])

        
        tmpSpecXR = getMultFrecSpec(dicSettings['wl'], mcTableTmp, dicSettings['velBins'],
                                    dicSettings['velCenterBin'], heightEdge1,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
                                    dicSettings['eps_diss'], dicSettings['uwind'], dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi, scatSet=dicSettings['scatSet'] )

        #volume normalization
        tmpSpecXR = tmpSpecXR/vol
        specXR = xr.merge([specXR, tmpSpecXR])
        
        if (dicSettings['scatSet']['mode'] == 'full') or (dicSettings['scatSet']['mode'] == 'table') or (dicSettings['scatSet']['mode'] == 'wisdom') :
            #calculating the integrated kdp
            tmpKdpXR = getIntKdp(dicSettings['wl'], mcTableTmp, heightEdge1)

            #volume normalization
            tmpKdpXR = tmpKdpXR/vol
        
            specXR = xr.merge([specXR, tmpKdpXR])
        
        counts[i] = len(mcTableTmp.vel.values)

    return specXR

