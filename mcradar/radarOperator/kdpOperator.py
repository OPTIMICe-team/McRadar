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
	tmpKdp: kdp calculated of a distribution of particles, separated for monomers and aggregates
	tmpKdp: dims=(range)
	"""

	tmpKdp = xr.Dataset()
	sKDP = mcTable.sKDP * mcTable['sMult']
	tmpKdp['KDP'] = sKDP.sum(dim='index').expand_dims({'range':np.asarray(centerHeight).reshape(1)})
	# now differently for Aggregates and Monomers
	mcTabledendrite = mcTable.where(mcTable['sPhi']<1,drop=True) # select only plates
	mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True)
	KDPMono = mcTabledendrite['sKDP'] * mcTabledendrite['sMult']
	tmpKdp['KDPMono'] = KDPMono.sum(dim='index').expand_dims({'range':np.asarray(centerHeight).reshape(1)})
	KDPAgg = mcTableAgg['sKDP'] * mcTableAgg['sMult']
	tmpKdp['KDPAgg'] = KDPAgg.sum(dim='index').expand_dims({'range':np.asarray(centerHeight).reshape(1)})
    
	return tmpKdp


