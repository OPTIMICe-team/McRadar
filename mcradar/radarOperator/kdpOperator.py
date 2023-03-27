# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import xarray as xr
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux


def getIntKdp(wls, elvs, mcTable, centerHeight):
    """
    Calculates the integrated kdp of a distribution 
    of particles.
    
    Parameters
    ----------
    mcTable: McSnow table returned from calcParticleKDP()
    centerHeight: height of the center of the distribution 
    of particles
    
    Returns
    -------
    kdpXR: kdp calculated of a distribution of particles
    kdbXR: dims=(range)
    """
    
    tmpKdp = xr.Dataset()
        
    for wl in wls:
        for elv in elvs:
            mcTabledendrite = mcTable[mcTable['sPhi']<1].copy() # select only plates
            mcTableAgg = mcTable[(mcTable['sNmono']>1)].copy()
            wlStr = '{:.2e}'.format(wl)
            mcTable['sKDPMult_{0}_elv{1}'.format(wlStr,elv)] = mcTable['sKDP_{0}_elv{1}'.format(wlStr,elv)] * mcTable['sMult']
            kdpXR = xr.DataArray(mcTable['sKDPMult_{0}_elv{1}'.format(wlStr,elv)].sum()[np.newaxis],
                                 dims=('range'),
                                 coords={'range':centerHeight[np.newaxis]},
                                 name='kdpInt_{0}_elv{1}'.format(wlStr,elv))
            
            tmpKdp = xr.merge([tmpKdp, kdpXR])
            #- now only dendrites
            mcTabledendrite['sKDPMult_{0}_elv{1}'.format(wlStr,elv)] = mcTabledendrite['sKDP_{0}_elv{1}'.format(wlStr,elv)] * mcTabledendrite['sMult']
            kdpXRMono = xr.DataArray(mcTabledendrite['sKDPMult_{0}_elv{1}'.format(wlStr,elv)].sum()[np.newaxis],
                                 dims=('range'),
                                 coords={'range':centerHeight[np.newaxis]},
                                 name='kdpIntMono_{0}_elv{1}'.format(wlStr,elv))
            
            tmpKdp = xr.merge([tmpKdp, kdpXR])
            
            mcTableAgg['sKDPMult_{0}_elv{1}'.format(wlStr,elv)] = mcTableAgg['sKDP_{0}_elv{1}'.format(wlStr,elv)] * mcTableAgg['sMult']
            kdpXRAgg = xr.DataArray(mcTableAgg['sKDPMult_{0}_elv{1}'.format(wlStr,elv)].sum()[np.newaxis],
                                 dims=('range'),
                                 coords={'range':centerHeight[np.newaxis]},
                                 name='kdpIntAgg_{0}_elv{1}'.format(wlStr,elv))
            
            tmpKdp = xr.merge([tmpKdp, kdpXR,kdpXRMono,kdpXRAgg])
    
    return tmpKdp


