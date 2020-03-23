# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import xarray as xr
from mcradar import *

def fullRadar(dicSettings, mcTable):


    specXR = xr.Dataset()
    counts = np.ones_like(dicSettings['heightRange'])*np.nan


    for i, heightEdge0 in enumerate(dicSettings['heightRange']):
    
        print(heightEdge0)
        heightEdge0, 
        centerHeight = heightEdge0 + dicSettings['heightRes']/2.
        heightEdge1 = heightEdge0 + dicSettings['heightRes']

        mcTableTmp = mcTable[(mcTable['sHeight']>=heightEdge0) & 
                             (mcTable['sHeight']<=heightEdge1)].copy()

        mcTableTmp = mcTableTmp[(mcTableTmp['sPhi']<=6)]

        #calculating doppler spectra
        mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], 
                                    mcTableTmp, ndgs=dicSettings['ndgsVal'])
    
        tmpSpecXR = getMultFrecSpec(dicSettings['wl'], mcTableTmp, dicSettings['velBins'], 
                                    dicSettings['velCenterBin'], centerHeight)
    
        specXR = xr.merge([specXR, tmpSpecXR])

        #calculating kdp
        mcTableTmp = calcParticleKDP(dicSettings['wl'], dicSettings['elv'], 
                                mcTableTmp, ndgs=dicSettings['ndgsVal'])
    
        tmpKdpXR = getIntKdp(dicSettings['wl'], mcTableTmp, centerHeight)
        specXR = xr.merge([specXR, tmpKdpXR])
    
        counts[i] = len(mcTableTmp.vel.values)


    return specXR

