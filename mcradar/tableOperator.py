# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import pandas as pd
import numpy as np


## it can be more general allowing the user to pass
## the name of the columns
def getMcSnowTable(mcSnowPath):
    """
    Read McSnow output table
    
    Parameters
    ----------
    mcSnowPath: path for the output from McSnow
    
    Returns
    -------
    Pandas DataFrame with the columns named after the local
    variable 'names'. This DataFrame additionally includes
    a column for the radii and the density [sRho]. The 
    velocity is negative towards the ground. 
    """
    
    names = ['time', 'mTot', 'sHeight', 'vel', 'dia', 
             'area', 'sMice', 'sVice', 'sPhi', 'sRhoIce',
             'igf', 'sMult', 'sMrime', 'sVrime']

    mcTable = pd.read_csv(mcSnowPath, header=None, names=names)
    selMcTable = mcTable.copy()
    selMcTable['vel'] = -1. * selMcTable['vel']
    selMcTable['radii_mm'] = selMcTable['dia']/2.
    selMcTable = calcRho(selMcTable)
            
    return selMcTable


def calcRho(mcTable):
    """
    Calculate the density of each super particles.
    
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    
    Returns
    -------
    mcTable with an additional column for the density.
    The density is calculated separately for aspect ratio < 1
    and for aspect ratio >= 1.
    """
    
    # density calculation for different AR ranges
    mcTable['sRho'] = np.ones_like(mcTable['time'])*np.nan

    #calculaiton for AR < 1
    tmpTable = mcTable[mcTable['sPhi']<1].copy()
    tmpVol = (np.pi/6.) * (tmpTable['dia']*1e2)**3 * tmpTable['sPhi']
    tmpRho = (tmpTable['mTot']*1e3)/tmpVol
    mcTable['sRho'].values[mcTable['sPhi']<1] = tmpRho

    # calculation for AR >= 1
    tmpTable = mcTable[mcTable['sPhi']>=1].copy()
    tmpVol = (np.pi/6.) * (tmpTable['dia']*1e2)**3 * tmpTable['sPhi']**2
    tmpRho = (tmpTable['mTot']*1e3)/tmpVol
    mcTable['sRho'].values[mcTable['sPhi']>=1] = tmpRho
    
    return mcTable


def creatZeCols(mcTable, wls):
    """
    Create the KDP column
    
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    wls: wavelenght (iterable) [mm]
    
    Returns
    -------
    mcTable with an empty columns 'sZe*_*' for 
    storing Ze_H and Ze_V of one particle of a 
    given wavelength
    """
    
    for wl in wls:
    
        wlStr = '{:.2e}'.format(wl)
        mcTable['sZeH_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sZeV_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan

        mcTable['sZeMultH_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sZeMultV_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan

    return mcTable


def creatKdpCols(mcTable, wls):
    """
    Create the KDP column
    
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    wls: wavelenght (iterable) [mm]
    
    Returns
    -------
    mcTable with an empty column 'sKDP_*' for 
    storing the calculated KDP of a given wavelength.
    """
    
    for wl in wls:
    
        wlStr = '{:.2e}'.format(wl)
        mcTable['sKDP_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
        mcTable['sKDPMult_{0}'.format(wlStr)] = np.ones_like(mcTable['time'])*np.nan
   
    return mcTable
