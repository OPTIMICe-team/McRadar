# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import xarray as xr
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar
from pytmatrix import refractive, tmatrix_aux


def getIntKdp(mcTable, centerHeight):
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

	sKDP = mcTable.sKPD * mcTableTmp['sMult']
	tmpKdp['KDP'] = sKDP.sum(dim='index').expand_dims({'range':centerHeight})
	# now differently for Aggregates and Monomers
	mcTabledendrite = mcTableTmp.where(mcTableTmp['sPhi']<1,drop=True) # select only plates
	mcTableAgg = mcTableTmp.where(mcTableTmp['sNmono']>1,drop=True)
	KDPMono = mcTabledendrite['sKDP'] * mcTabledendrite['sMult']
	tmpKdp['KDPMono'] = KDPMono.sum(dim='index').expand_dims({'range':centerHeight})
	KDPAgg = mcTableAgg['sKDP'] * mcTableAgg['sMult']
	tmpKdp['KDPAgg'] = KDPAgg.sum(dim='index').expand_dims({'range':centerHeight})
	
	#for wl in wls:

	#    wlStr = '{:.2e}'.format(wl)
	#    mcTable['sKDPMult_{0}'.format(wlStr)] = mcTable['sKDP_{0}'.format(wlStr)] * mcTable['sMult']
	#    kdpXR = xr.DataArray(mcTable['sKDPMult_{0}'.format(wlStr)].sum()[np.newaxis],
	#                         dims=('range'),
	#                         coords={'range':centerHeight[np.newaxis]},
	#                         name='kdpInt_{0}'.format(wlStr))

	#    tmpKdp = xr.merge([tmpKdp, kdpXR])
    
    return tmpKdp


