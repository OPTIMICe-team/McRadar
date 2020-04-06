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
    counts = np.ones_like(dicSettings['heightRange'])*np.nan
    vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']

    for i, heightEdge0 in enumerate(dicSettings['heightRange']):

        heightEdge1 = heightEdge0 + dicSettings['heightRes']

        print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))

        mcTableTmp = mcTable[(mcTable['sHeight']>heightEdge0) &
                             (mcTable['sHeight']<=heightEdge1)].copy()

        mcTableTmp = mcTableTmp[(mcTableTmp['sPhi']<=6)]

        #calculating Ze of each particle
        mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
                                    mcTableTmp, ndgs=dicSettings['ndgsVal'])

        #calculating doppler spectra
        tmpSpecXR = getMultFrecSpec(dicSettings['wl'], mcTableTmp, dicSettings['velBins'],
                                    dicSettings['velCenterBin'], heightEdge1)

        #volume normalization
        tmpSpecXR = tmpSpecXR/vol
        specXR = xr.merge([specXR, tmpSpecXR])

        #calculating kdp of each particle
        mcTableTmp = calcParticleKDP(dicSettings['wl'], dicSettings['elv'],
                                mcTableTmp, ndgs=dicSettings['ndgsVal'])

        #calculating the integrated kdp
        tmpKdpXR = getIntKdp(dicSettings['wl'], mcTableTmp, heightEdge1)

        #volume normalization
        tmpKdpXR = tmpKdpXR/vol

        specXR = xr.merge([specXR, tmpKdpXR])

        counts[i] = len(mcTableTmp.vel.values)

    return specXR

