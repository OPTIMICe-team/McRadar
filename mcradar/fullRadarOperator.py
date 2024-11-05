# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: José Dias Neto


import xarray as xr
from mcradar import *
import matplotlib.pyplot as plt
from mcradar.tableOperator import creatRadarCols
import time
import multiprocessing
from multiprocessing import Process, Queue
import sys
if not sys.warnoptions: # bad form, would not recommend!
    import warnings
    warnings.simplefilter("ignore")
debugging=True
reduce_ncores = True
def getRadarParParallel(heightEdge0,mcTable,dicSettings):#heightRes,wl,elv,ndgsVal,scatSet,velBins,velCenterBin,convolute,nave,noise_pow,eps_diss,uwind,time_int,theta,tau):
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	heightEdge1 = heightEdge0 + dicSettings['heightRes']

	print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
	mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
			 					(mcTable['sHeight']<=heightEdge1),drop=True)
	
	if mcTableTmp.vel.any():
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
					       			mcTableTmp, ndgs=dicSettings['ndgsVal'],
					        		scatSet=dicSettings['scatSet'])#,height=(heightEdge1+heightEdge0)/2)
		k_theta, k_phi, k_r = 0,0,0
		tmpSpecXR = getMultFrecSpec(dicSettings['wl'], dicSettings['elv'],mcTableTmp, dicSettings['velBins'],
					        		dicSettings['velCenterBin'], (heightEdge1+heightEdge0)/2,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
					        		dicSettings['eps_diss'], dicSettings['uwind'],dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi,
					        		k_theta,k_phi,k_r, dicSettings['tau'],
					        		scatSet=dicSettings['scatSet'])


		
		tmpSpecXR = tmpSpecXR/vol
	
		if (dicSettings['scatSet']['mode'] == 'full') or (dicSettings['scatSet']['mode'] == 'table') or (dicSettings['scatSet']['mode'] == 'wisdom') or (dicSettings['scatSet']['mode'] == 'DDA'):
		#calculating the integrated kdp
			tmpKdpXR =  getIntKdp(mcTableTmp,(heightEdge1+heightEdge0)/2)
			tmpSpecXR = xr.merge([tmpSpecXR, tmpKdpXR/vol])
			#print(specXR)
		#print(tmpSpecXR)
		#plt.plot(tmpSpecXR.vel,tmpSpecXR.sel(elevation=90,wavelength=31.23,range=tmpSpecXR.range[0],method='nearest').spec_H)
		#plt.savefig('/project/meteo/work/L.Terzi/McSnow_habit/test_McRadar_height{}.png'.format(tmpSpecXR.range.values))
		#plt.close()
		#quit()		
		return tmpSpecXR
	else:
		print('empty dataset at this height range')


	

def fullRadarParallel(dicSettings, mcTable):
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
	
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	n_cores = 4#multiprocessing.cpu_count()
	#if reduce_ncores:
	#	if n_cores > 1:
	#		n_cores = n_cores - 1 # we have the main function running on one core and our institute does not allow to use all cores
	print(n_cores)
	pool = multiprocessing.Pool(n_cores)
	
	args = [(h, mcTable, dicSettings) for h in dicSettings['heightRange']]
	
	#for result in pool.starmap(getRadarParParallel,args):
	#for h in dicSettings['heightRange']:
	#	result = getRadarParParallel(h,mcTable,dicSettings)
		#print(result)
		#quit()
	#	if result:
	#		specXR = xr.merge([specXR,result])
	result =  pool.starmap(getRadarParParallel,args)
	result = [x for x in result if x is not None]
	
	#print(result)
	specXR = xr.merge(result)
	
	if debugging:
		print('total time with parallelizing for all heights was', time.time()-t0)
	
	return specXR



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
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	for i, heightEdge0 in enumerate(dicSettings['heightRange']):

		heightEdge1 = heightEdge0 + dicSettings['heightRes']

		print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
		mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
				 					(mcTable['sHeight']<=heightEdge1),drop=True)
		
		if mcTableTmp.vel.any():
			mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
						       			mcTableTmp, ndgs=dicSettings['ndgsVal'],
						        		scatSet=dicSettings['scatSet'])#,height=(heightEdge1+heightEdge0)/2)
			#quit()
			if (heightEdge0 >= dicSettings['shear_height0']) and (heightEdge1 <= dicSettings['shear_height1']): # only if we are within the shear zone, have shear! TODO make it possible to have profile of wind shear read in!!
				k_theta,k_phi,k_r = dicSettings['k_theta'], dicSettings['k_phi'], dicSettings['k_r']
				
			else:
				k_theta, k_phi, k_r = 0,0,0
			tmpSpecXR = getMultFrecSpec(dicSettings['wl'], dicSettings['elv'],mcTableTmp, dicSettings['velBins'],
						        		dicSettings['velCenterBin'], (heightEdge1+heightEdge0)/2,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
						        		dicSettings['eps_diss'], dicSettings['uwind'],dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi,
						        		k_theta,k_phi,k_r, dicSettings['tau'],
						        		scatSet=dicSettings['scatSet'])


			#volume normalization
			#for var in tmpSpecXR:
		#		print(var)
		#		if 'Broad' not in var:
		#			tmpSpecXR[var] = tmpSpecXR[var]/vol
			tmpSpecXR = tmpSpecXR/vol
			#plt.plot(tmpSpecXR.vel,10*np.log10(tmpSpecXR.spec_H.sel(elevation=90,wavelength=3.189,range=(heightEdge1+heightEdge0)/2,method='nearest')))
			#print(10*np.log10(tmpSpecXR.spec_H.sum(dim='vel')/tmpSpecXR.spec_V.sum(dim='vel')))
			#plt.show()
			

			#print(tmpSpecXR)
			##quit()
			specXR = xr.merge([specXR, tmpSpecXR])
			
			if dicSettings['attenuation'] == True:
				tmpAtt = get_attenuation(mcTableTmp, dicSettings['wl'], dicSettings['elv'],
																			dicSettings['temp'].sel(range=(heightEdge1+heightEdge0)/2), dicSettings['relHum'].sel(range=(heightEdge1+heightEdge0)/2), dicSettings['press'].sel(range=(heightEdge1+heightEdge0)/2),
																			dicSettings['scatSet']['mode'],	vol,(heightEdge1+heightEdge0)/2,dicSettings['heightRes'])#,att_atm0,att_ice_HH0,att_ice_VV0) 			
				
				specXR = xr.merge([specXR,tmpAtt])
				#print(specXR)
				#quit()
				
		
			if (dicSettings['scatSet']['mode'] == 'full') or (dicSettings['scatSet']['mode'] == 'table') or (dicSettings['scatSet']['mode'] == 'wisdom') or (dicSettings['scatSet']['mode'] == 'DDA'):
			#calculating the integrated kdp
				tmpKdpXR =  getIntKdp(mcTableTmp,(heightEdge1+heightEdge0)/2)
				specXR = xr.merge([specXR, tmpKdpXR/vol])
				#print(specXR)
					
		
		else:
			print('empty dataset at this height range')
	
	if dicSettings['attenuation'] == True:
		#print(2*np.cumsum(specXR.att_atmo.cumsum(dim='range')))
		#plt.plot(specXR.att_atmo.sel(wavelength=specXR.wavelength[0],elevation=90),specXR.range,label='atmo')
		#plt.plot(specXR.att_ice_HH.sel(wavelength=specXR.wavelength[0],elevation=90),specXR.range,label='ice')
		#plt.savefig('test_att_atmo_dh.png')
		#plt.show()
		#quit()
		specXR['att_atm_ice_HH'] = 2*specXR.att_ice_HH.cumsum(dim='range') + 2*specXR.att_atmo.cumsum(dim='range')
		specXR['att_atm_ice_VV'] = 2*specXR.att_ice_VV.cumsum(dim='range') + 2*specXR.att_atmo.cumsum(dim='range')
		specXR.att_atm_ice_HH.attrs['long_name'] = '2 way attenuation at HH polarization'
		specXR.att_atm_ice_HH.attrs['unit'] = 'dB'
		specXR.att_atm_ice_HH.attrs['comment'] = '2 way attenuation for ice particles and atmospheric gases (N2,O2,H2O). The spectra are divided my this, so to get unattenuated spectra, multiply with this (in linear units)'
		
		specXR.att_atm_ice_HH.attrs['long_name'] = '2 way attenuation at VV polarization'
		specXR.att_atm_ice_HH.attrs['unit'] = 'dB'
		specXR.att_atm_ice_HH.attrs['comment'] = '2 way attenuation for ice particles and atmospheric gases (N2,O2,H2O). The spectra are divided my this, so to get unattenuated spectra, multiply with this (in linear units)'
		
		if (dicSettings['scatSet']['mode'] == 'SSRGA') or (dicSettings['scatSet']['mode'] == 'Rayleigh') or (dicSettings['scatSet']['mode'] == 'SSRGA-Rayleigh'):
			specXR['spec_H'] = specXR.spec_H/(10**(specXR.att_atm_ice_HH/10))
		else:		
			specXR['spec_H_att'] = specXR.spec_H/(10**(specXR.att_atm_ice_HH/10))
			specXR['spec_V'] = specXR.spec_V/(10**(specXR.att_atm_ice_VV/10))
			specXR['spec_HV'] = specXR.spec_HV/(10**(specXR.att_atm_ice_HH/10))
		
	if debugging:
		print('total time for all heights was', time.time()-t0)
	return specXR

def singleParticleTrajectories(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the single particle scattering properties
	"""

	t0 = time.time()
	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	mcTable = creatRadarCols(mcTable, dicSettings)
	#counts = np.ones_like(dicSettings['heightRange'])*np.nan
	#vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']

	for i, pID in enumerate(np.unique(mcTable['sMult'].values)):

		mcTableTmp = mcTable.where(mcTable.sMult==pID,drop=True)
		print(len(np.unique(mcTable['sMult'].values)),i)
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
									mcTableTmp, ndgs=dicSettings['ndgsVal'],
									scatSet=dicSettings['scatSet'])
		
		mcTableTmp = mcTableTmp.assign_coords(index=mcTableTmp.sHeight).rename({'index':'range'})#.set_index('sHeight')
		mcTableTmp = mcTableTmp.reindex(range=dicSettings['heightRange'],method='nearest',tolerance=dicSettings['heightRes'])
		vars2drop = ['sMult','sZeMultH','sZeMultV','sZeMultHV','sCextHMult','sCextHMult','sKDPMult']
		mcTableTmp = mcTableTmp.drop_vars(vars2drop)
		mcTableTmp = mcTableTmp.expand_dims(dim='pID').assign_coords(pID=[pID])
		#print(mcTableTmp)
		#quit()
		specXR = xr.merge([specXR, mcTableTmp])
		
	print('total time was ', time.time()-t0)
	return specXR

def calc_1_particle(mcTable,pID,dicSettings,i):
	mcTableTmp = mcTable.where(mcTable.sMult==pID,drop=True)
	print(len(np.unique(mcTable['sMult'].values)[::2]),i)
	mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
								mcTableTmp, ndgs=dicSettings['ndgsVal'],
								scatSet=dicSettings['scatSet'])
	mcTableTmp = mcTableTmp.assign_coords(index=mcTableTmp.sHeight).rename({'index':'range'})#.set_index('sHeight')
	mcTableTmp = mcTableTmp.reindex(range=dicSettings['heightRange'],method='nearest',tolerance=dicSettings['heightRes'])
	vars2drop = ['sMult','sZeMultH','sZeMultV','sZeMultHV','sCextHMult','sCextHMult','sKDPMult']
	mcTableTmp = mcTableTmp.drop_vars(vars2drop)
	
	mcTableTmp = mcTableTmp.expand_dims(dim='pID').assign_coords(pID=[pID])
	return mcTableTmp

def singleParticleTrajParallel(dicSettings, mcTable,MaxWorkers=10):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the single particle scattering properties
	"""

	t0 = time.time()
	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	mcTable = creatRadarCols(mcTable, dicSettings)
	#counts = np.ones_like(dicSettings['heightRange'])*np.nan
	#vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']

	pool = multiprocessing.Pool(MaxWorkers)
	
	args = [(mcTable, pID, dicSettings,i) for i,pID in enumerate(np.unique(mcTable['sMult'].values)[::5])]
	
	result =  pool.starmap(calc_1_particle,args)
	print('done with calcs, now need to merge')
	result = [x for x in result if x is not None]
	print(result)
	specXR = xr.merge(result)
		
	print('total time was ', time.time()-t0)
	return specXR


def singleParticleScat(dicSettings, mcTable):
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

	t0 = time.time()
	singlePart = xr.Dataset()
	#specXR_turb = xr.Dataset()
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	for i, heightEdge0 in enumerate(dicSettings['heightRange']):

		heightEdge1 = heightEdge0 + dicSettings['heightRes']

		print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
		mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
				 					(mcTable['sHeight']<=heightEdge1),drop=True)
		
		if mcTableTmp.vel.any():
			mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
						       			mcTableTmp, ndgs=dicSettings['ndgsVal'],
						        		scatSet=dicSettings['scatSet'])#,height=(heightEdge1+heightEdge0)/2)
			#print(mcTableTmp.sZeH)
			singlePart = xr.merge([mcTableTmp,singlePart]) # TODO: do I need to normalize with Volume? I think so!!
			#print(singlePart.sZeH)
	print('total time with old method', time.time()-t0)
	return singlePart


'''
def singleParticleTrajectories(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the single particle scattering properties
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
		#for i, pID in enumerate(mcTable['sMult'].unique()):

		#    mcTableTmp = mcTable[(mcTable['sMult']==pID)].copy()

		#print(len(mcTable['sMult'].unique()),i)
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
				                    mcTableTmp, ndgs=dicSettings['ndgsVal'],
				                    scatSet=dicSettings['scatSet'])
		print(mcTableTmp)
		quit()
		mcTableTmp = mcTableTmp.set_index('sHeight')
		specTable = mcTableTmp.to_xarray()
		specTable = specTable.drop_vars('sMult')
		specTable = specTable.expand_dims(dim='sMult').assign_coords(sMult=[pID])
		print(specTable)

		#specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
		specXR = xr.merge([specXR, specTable])
		print(specXR)
		quit()

	return specXR
'''
