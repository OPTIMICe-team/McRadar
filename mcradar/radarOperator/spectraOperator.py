# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pandas as pd


def getVelIntSpec(mcTable, mcTable_binned, variable):
    """
    Calculates the integrated reflectivity for each velocity bin
    
    Parameters
    ----------
    mcTable: McSnow output returned from calcParticleZe()
    mcTable_binned: McSnow table output binned for a given velocity bin
    variable: name of column variable wich will be integrated over a velocity bin
    
    Returns
    -------
    mcTableVelIntegrated: table with the integrated reflectivity for each velocity bin
    """

    mcTableVelIntegrated = mcTable.groupby(mcTable_binned)[variable].agg(['sum'])
    
    return mcTableVelIntegrated


def getMultFrecSpec(wls, mcTable, velBins, velCenterBins , centerHeight):
    """
    Calculation of the multi-frequency spectrograms 
    
    Parameters
    ----------
    wls: wavelenght (iterable) [mm]
    mcTable: McSnow output returned from calcParticleZe()
    velBins: velocity bins for the calculation of the spectrogram (array) [m/s]
    velCenterBins: center of the velocity bins (array) [m/s]
    centerHeight: center height of each range gate (array) [m]
    
    Returns
    -------
    xarray dataset with the multi-frequency spectrograms
    xarray dims = (range, vel) 
    """
    
    mcTable_binned = pd.cut(mcTable['vel'], velBins)

    tmpDataDic = {}
     
    for wl in wls:
    
        wlStr = '{:.2e}'.format(wl)
	
        mcTable['sZeMultH_{0}'.format(wlStr)] = mcTable['sZeH_{0}'.format(wlStr)] * mcTable['sMult']
        mcTable['sZeMultV_{0}'.format(wlStr)] = mcTable['sZeV_{0}'.format(wlStr)] * mcTable['sMult']

        intSpecH = getVelIntSpec(mcTable, mcTable_binned,'sZeMultH_{0}'.format(wlStr))
        tmpDataDic['spec_H_{0}'.format(wlStr)] = intSpecH.values[:,0]

        intSpecV = getVelIntSpec(mcTable, mcTable_binned, 'sZeMultV_{0}'.format(wlStr))
        tmpDataDic['spec_V_{0}'.format(wlStr)] = intSpecV.values[:,0]

    #converting to XR
    specTable = pd.DataFrame(data=tmpDataDic, index=velCenterBins)
    specTable = specTable.to_xarray()
    specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
    specTable = specTable.rename_dims(dims_dict={'index':'vel'}).rename(name_dict={'index':'vel'})
    
    return specTable
