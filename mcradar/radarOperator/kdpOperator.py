# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import xarray as xr
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux


def getIntKdp(wls, mcTable, centerHeight):
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
        
        wlStr = '{:.2e}'.format(wl)
        mcTable['sKDPMult_{0}'.format(wlStr)] = mcTable['sKDP_{0}'.format(wlStr)] * mcTable['sMult']
        
        kdpXR = xr.DataArray(mcTable['sKDPMult_{0}'.format(wlStr)].sum()[np.newaxis],
                             dims=('range'),
                             coords={'range':centerHeight[np.newaxis]},
                             name='kdpInt_{0}'.format(wlStr))
        
        tmpKdp = xr.merge([tmpKdp, kdpXR])
    
    return tmpKdp


