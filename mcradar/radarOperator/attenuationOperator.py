import subprocess
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from scipy import constants
import warnings
import matplotlib.pyplot as plt
import time
import multiprocessing
from multiprocessing import Process, Queue
#import pyPamtra

def get_attenuation(mcTable,wls,elvs,temp,relHum,press,mode,vol,centerHeight,heightRes):#,att_atm0,att_ice_HH0,atm_att_ice_VV0):
	'''
	calculate 2 way attenutation due to N2, O2, Water vapour using Pamtra. 
	Parameters
	---------- 
	tmpAtt: McSnow output returned from calcParticleZe()
	atmoFile: atmofile generated in McSnow output
	dicSettings: the dicSettings defined in settings.py
	
	Returns
	---------
	output: calculated attenuation
	'''
	#output['spec_H_att'] = output.spec_H.copy()
	tmpAtt = xr.Dataset()
	Att_array = xr.DataArray(data = np.zeros((len(elvs),len(wls))),
							dims=['elevation','wavelength'], coords={'elevation':elvs,'wavelength':wls},
		       				attrs={'long_name':'attenuation due to atmospheric gases in height bin',
		               				'units':r'dBm$^{-1}$'})
	tmpAtt['att_atmo'] = Att_array
	#print(mcTable)
	#quit()
	Att_array = xr.DataArray(data = np.zeros((len(elvs),len(wls))),
							dims=['elevation','wavelength'], coords={'elevation':elvs,'wavelength':wls},
		        			attrs={'long_name':'attenuation due to ice particles at HH polarization in height bin',
		               				'units':r'dBm$^{-1}$'})
	tmpAtt['att_ice_HH'] = Att_array
	
	Att_array = xr.DataArray(data = np.zeros((len(elvs),len(wls))),
							dims=['elevation','wavelength'], coords={'elevation':elvs,'wavelength':wls},
		        			attrs={'long_name':'attenuation due to ice particles at VV polarization in height bin',
		               				'units':r'dBm$^{-1}$'})
	tmpAtt['att_ice_VV'] = Att_array
	
	
	for wl in wls:
		freq = 299792458e3/wl
		for elv in elvs:
			#- calculate attenuation
			#att = getAtmAttPamtra(output.range,atmoReindex.temp,atmoReindex.relHum,atmoReindex.press,dicSettings['freq']*1e-9)
			#attAtm=att['Att_atmo'][:,0,:, 0]
			if (mode == 'SSRGA') or (mode == 'Rayleigh') or (mode == 'SSRGA-Rayleigh'):		
				
				att_atm, att_ice_HH = getHydroAtmAtt(temp,relHum,press,dicSettings['freq']*1e-9,dicSettings['heightRes'],VV=False)
				attAtm2Way = 2*np.cumsum(attAtm,axis=1)
				attIce2Way_HH = 2*np.cumsum(att_ice_HH,axis=1)
			
				tmpAtt['att_atm'].loc[:,elv,wl] = attAtm2Way
				tmpAtt['att_ice_HH'].loc[:,elv,wl] = attIce2Way_HH
				tmpAtt['att_atm_ice_HH'].loc[:,elv,wl] = attAtm2Way + attIce2Way_HH
			else:
				
				# now calculate attenuation due to ice particles
				mcTable['sCextHMult'] = mcTable['sCextH'] * mcTable['sMult']
				mcTable['sCextVMult'] = mcTable['sCextV'] * mcTable['sMult']
				
				sCext_H_sum = mcTable['sCextHMult'].sum(dim='index')/vol/mcTable.sMult.sum()
				sCext_V_sum = mcTable['sCextVMult'].sum(dim='index')/vol/mcTable.sMult.sum()
				
				tmpAtt['att_ice_HH']  = 10*np.log10(np.exp(sCext_H_sum))# * heightRes))#*4.343e-3 
				tmpAtt['att_ice_VV']  = 10*np.log10(np.exp(sCext_V_sum))# * heightRes))#*4.343e-3 #10*np.log10(np.exp(kexthydro*delta_h))
				#attIce2Way_VV = #2*np.cumsum(att_ice_VV,axis=1)
				#attIce2Way_HH = #2*np.cumsum(att_ice_HH,axis=1)
				
				#- get atmospheric attenuation				
				rt_kextatmo = getHydroAtmAtt(temp,relHum,press,freq*1e-9)
				tmpAtt['att_atmo'].loc[elv,wl] = 10*np.log10(np.exp(rt_kextatmo.values * heightRes))
				
				#attAtm2Way = 2*np.cumsum(attAtm,axis=1)
				
				#tmpAtt['att_atm'].loc[:,elv,wl] = attAtm2Way
				#tmpAtt['att_ice_HH'].loc[:,elv,wl] = attIce2Way_HH
				#tmpAtt['att_ice_VV'].loc[:,elv,wl] = attIce2Way_VV
				#tmpAtt['att_atm_ice_HH'].loc[:,elv,wl] = attAtm2Way + attIce2Way_HH
				#tmpAtt['att_atm_ice_VV'].loc[:,elv,wl] = attAtm2Way + attIce2Way_VV
	
	tmpAtt = tmpAtt.expand_dims(dim='range').assign_coords(range=[centerHeight])
	return tmpAtt

def getHydroAtmAtt(temp,relHum,pres,freq):
	'''
	calculate attenuation how it is done in PAMTRA. It needs to have getAtmAttenuation
	Parameters
	----------
	temp: temperature in Kelvin
	relHum: relative humidity with respect to liquid water %
	freq: the frequency of the radar in GHz
	pres: pressure in Pa
	Returns
	rt_kextatmo: extinction coeff. of atmosphere. To get attenuation in dB: 10*np.log10(np.exp(rt_kextatmo.values * dZ)) with dZ height increment
	----------
	'''
	r_v = 461.5249933083879
	atmo_vap_pressure = relHum * e_sat_gg_water(temp)
	#atmo_vapor_pressure(nx,ny,nz) = atmo_relhum(nx,ny,nz) * e_sat_gg_water(atmo_temp(nx,ny,nz)) 

	atmo_rho_vap = atmo_vap_pressure/(temp*r_v) # TODO define r_v
	#atmo_rho_vap(i,j,lay_use+ii)        = atmo_vapor_pressure(i,j,lay_use+ii)/(atmo_temp(i,j,lay_use+ii) * r_v)

	absair,abswv = getAtmAttenuation(temp,freq,atmo_rho_vap,pres)

	rt_kextatmo = (absair + abswv)/1e3    # conversion to Np/m

	return rt_kextatmo

#PIA = 2 * (SUM(out_att_hydro(i_x,i_y,1:i_z-1,i_f,1)) + SUM(out_att_atmo(i_x,i_y,1:i_z-1,i_f))) # basically cumsum of out_att_hydro and out_att_atmo. What we have here is in dB
#PIA = PIA + out_att_hydro(i_x,i_y,i_z,i_f,1) + out_att_atmo(i_x,i_y,i_z,i_f)


def e_sat_gg_water(T):
	'''    
	Calculates the saturation pressure over water after Goff and Gratch (1946).
	It is the most accurate that you can get for a temperture range from -90°C to +80°C.
	Source: Smithsonian Tables 1984, after Goff and Gratch 1946
	http://cires.colorado.edu/~voemel/vp.html
	http://hurri.kean.edu/~yoh/calculations/satvap/satvap.html
	Parameters
	----------
	T: Temperature in Kelvin
	Returns
	----------
	e_sat_gg_water: saturation pressure over water in hPa
	'''

	e_sat_gg_water = 1013.246 * 10**( -7.90298*(373.16/T-1) + 5.02808*np.log10(373.16/T) - 1.3816e-7*(10**(11.344*(1-T/373.16))-1) + 8.1328e-3 * (10**(-3.49149*(373.16/T-1))-1) )

	return e_sat_gg_water
	
def getAtmAttenuation(temp,freq,rhoWv,pres):
	'''
	calculated gas attenuation accoring to Rosenkranz 98 model. This is based on pamtra
	Based on frequency, temperature, water vapor density, and pressure, this routine
    calculates the absorption due to air (N2 and O2) and water vapor in the frequency
    range from 0.1 to 800.0 GHz, pressure range from 10 to 1.2e5 Pa, and absolute
    temperatures larger than 100 K.
    
    Parameters
    ----------
    temp: temperature in Kelvin
    freq: frequency of radar in GHz
    rhoWv: water vapour density in kg/m**3
    pres: pressure in Pa
    Returns
    ----------
    absAir: extiction by dry air in Np/km
    absWv: extinction by water vapour in Np/km
	'''
	
	# convert pressure from Pa to Mb
	pmb = pres / 100.0
	
	# convert vapor density from kg/m**3 to g/m**3
	vapden = rhoWv * 1000.0
	
	# get volume extinction coefficients
	absAir = absn2(temp,pmb,freq) + abso2(temp,pmb,vapden,freq)
	absWv = absh2o(temp,pmb,vapden,freq)
	
	return absAir,absWv
	
	
def absn2(temp,pres,freq):
	'''
	calculate absorption of N2 gas
	Based on PAMTRA
	
	Parameters
	----------
    temp: temperature in Kelvin
    freq: frequency of radar in GHz
    pres: pressure in Mb
    Returns
    ----------
	absn2 = absorption coefficient due to nitrogen in air [NEPER/KM]
	'''
	
	th = 300./temp
	absn2 = 6.4e-14*pres**2*freq**2*th**3.55
	return absn2
	
def abso2(tempK,pres,vapden,freq):
	''' 
	calculate absorption of O2 gas
	Based on PAMTRA
	
	Parameters
	----------
    tempK: temperature in Kelvin
    freq: frequency of radar in GHz
    pres: pressure in Mb
    vapden: varpour density in g/m**3
    Returns
    ----------
	abso2 = absorption coefficient due to oxygen in air [NEPER/KM]
	'''
	
	x=0.8; wb300=0.56
	
	w300 = np.array([1.63, 1.646, 1.468, 1.449, 1.382, 1.360,
          			 1.319, 1.297, 1.266, 1.248, 1.221, 1.207, 1.181, 1.171,
         			 1.144, 1.139, 1.110, 1.108, 1.079, 1.078, 1.05, 1.05,
          			 1.02, 1.02, 1.0, 1.0, 0.97, 0.97, 0.94, 0.94, 0.92, 0.92, 
          			 0.89, 0.89, 1.92, 1.92, 1.92, 1.81, 1.81, 1.81])
	y300 = np.array([-0.0233,  0.2408, -0.3486,  0.5227,
          			 -0.5430,  0.5877, -0.3970,  0.3237, -0.1348,  0.0311,
         			 0.0725, -0.1663,  0.2832, -0.3629,  0.3970, -0.4599,
         			 0.4695, -0.5199,  0.5187, -0.5597,  0.5903, -0.6246,
         			 0.6656, -0.6942,  0.7086, -0.7325,  0.7348, -0.7546,
         			 0.7702, -0.7864,  0.8083, -0.8210,  0.8439, -0.8529,
         			 0., 0., 0., 0., 0., 0.])
	v = np.array([0.0079, -0.0978,  0.0844, -0.1273,
				   0.0699, -0.0776,  0.2309, -0.2825,  0.0436, -0.0584,
				   0.6056, -0.6619,  0.6451, -0.6759,  0.6547, -0.6675,
				   0.6135, -0.6139,  0.2952, -0.2895,  0.2654, -0.2590,
				   0.3750, -0.3680,  0.5085, -0.5002,  0.6206, -0.6091,
				   0.6526, -0.6393,  0.6640, -0.6475,  0.6729, -0.6545,
				   0., 0., 0., 0., 0., 0.])
	f = np.array([118.7503, 56.2648, 62.4863, 58.4466, 60.3061, 59.5910,
				  59.1642, 60.4348, 58.3239, 61.1506, 57.6125, 61.8002,
				  56.9682, 62.4112, 56.3634, 62.9980, 55.7838, 63.5685,
				  55.2214, 64.1278, 54.6712, 64.6789, 54.1300, 65.2241,
				  53.5957, 65.7648, 53.0669, 66.3021, 52.5424, 66.8368,
				  52.0214, 67.3696, 51.5034, 67.9009, 368.4984, 424.7631,
				  487.2494, 715.3932, 773.8397, 834.1453])
	s300 = np.array([0.2936E-14, 0.8079E-15, 0.2480E-14, 0.2228E-14,
					 0.3351E-14, 0.3292E-14, 0.3721E-14, 0.3891E-14,
					 0.3640E-14, 0.4005E-14, 0.3227E-14, 0.3715E-14,
					 0.2627E-14, 0.3156E-14, 0.1982E-14, 0.2477E-14,
					 0.1391E-14, 0.1808E-14, 0.9124E-15, 0.1230E-14,
					 0.5603E-15, 0.7842E-15, 0.3228E-15, 0.4689E-15,
					 0.1748E-15, 0.2632E-15, 0.8898E-16, 0.1389E-15,
					 0.4264E-16, 0.6899E-16, 0.1924E-16, 0.3229E-16,
					 0.8191E-17, 0.1423E-16, 0.6460E-15, 0.7047E-14,
					 0.3011E-14, 0.1826E-14, 0.1152E-13, 0.3971E-14])
	be = np.array([0.009, 0.015, 0.083, 0.084, 0.212, 0.212, 0.391, 0.391, 0.626, 0.626,
				   0.915, 0.915, 1.26, 1.26, 1.66, 1.665, 2.119, 2.115, 2.624, 2.625,
				   3.194, 3.194, 3.814, 3.814, 4.484, 4.484, 5.224, 5.224, 6.004, 6.004,
				   6.844, 6.844,7.744, 7.744, 0.048, 0.044, 0.049, 0.145, 0.141, 0.145])
	th = 300./tempK
	th1 = th-1.
	b = th**x
	preswv = vapden*tempK/217.
	presda = pres-preswv
	den = 0.001*(presda*b + 1.1*preswv*th)
	dfnr = wb300*den
	loc_sum = 1.6e-17*freq**2*dfnr/(th*(freq**2 + dfnr**2))
	
	for k in range(40):#do k=1,40
		df = w300[k]*den
		y = 0.001*pres*b*(y300[k]+v[k]*th1)
		strk = s300[k]*np.exp(-be[k]*th1)
		sf1 = (df + (freq-f[k])*y)/((freq-f[k])**2 + df**2)
		sf2 = (df - (freq+f[k])*y)/((freq+f[k])**2 + df**2)
		loc_sum = loc_sum + strk*(sf1+sf2)*(freq/f[k])**2
	#end do
	#print(loc_sum)
	loc_sum = 1.6e-17*freq**2*dfnr/(th*(freq**2 + dfnr**2))
	'''
	n_cores = multiprocessing.cpu_count()
	if n_cores > 1:
		n_cores = n_cores - 1
	pool = multiprocessing.Pool(n_cores)
	args = [(w300s,den,pres,b,y300s,vs,th1,s300s,bes,freq,fs) for w300s,y300s,vs,s300s,bes,fs in zip(w300,y300,v,s300,be,f)]
	for result in pool.starmap(forLoopO2,args):
		if result:
			loc_sum = loc_sum + result
	#print(loc_sum)
	#quit()
	'''
	
	o2abs = 0.5034e12*loc_sum*presda*th**3/np.pi
	return o2abs

def forLoopO2(w300,den,pres,b,y300,v,th1,s300,be,freq,f):
	df = w300*den
	y = 0.001*pres*b*(y300+v*th1)
	strk = s300*np.exp(-be*th1)
	sf1 = (df + (freq-f)*y)/((freq-f)**2 + df**2)
	sf2 = (df - (freq+f)*y)/((freq+f)**2 + df**2)
	return strk*(sf1+sf2)*(freq/f)**2
	
def absh2o(tempK,pres,rho,freq):
	'''
	calculated absorption due to water vapour
	Parameters
	----------
	tempK: temperature in Kelvin
	pres: pressure in millibar
	rho: water vapour density [g/m**3]
	freq: frequency in GHz
	Returns
	----------
	absh2o: absorption due to water vapour in nepers/km
	'''
	
	if(rho <= 0.):
		absh20 = 0.
		return asbh2o
	
	else:
		fl = np.array([22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508,
		      		   443.0183, 448.0011, 470.8890, 474.6891, 488.4911, 556.9360,
		      		    620.7008, 752.0332, 916.1712])
		s1 = np.array([0.1310E-13, 0.2273E-11, 0.8036E-13, 0.2694E-11, 0.2438E-10,
		    		   0.2179E-11, 0.4624E-12, 0.2562E-10, 0.8369E-12, 0.3263E-11,
				       0.6659E-12, 0.1531E-08, 0.1707E-10, 0.1011E-08, 0.4227E-10])
		b2 = np.array([2.144, .668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405,
				       3.597, 2.379, 2.852, .159, 2.391, .396, 1.441])
		w3 = np.array([0.002656, 0.00281, 0.0023, 0.00278, 0.00287, 0.0021, 0.00186,
				       0.00263, 0.00215, 0.00236, 0.0026, 0.00321, 0.00244, 0.00306, 0.00267])
		x = np.array([0.69, 0.64, 0.67, 0.68, 0.54, 0.63, 0.60, 0.66, 0.66,
				      0.65, 0.69, 0.69, 0.71, 0.68, 0.70])
		ws = np.array([0.0127488, 0.01491, 0.0108, 0.0135, 0.01541, 0.0090, 0.00788,
		      		   0.01275, 0.00983, 0.01095, 0.01313, 0.01320, 0.01140, 0.01253, 0.01275])
		xs = np.array([0.61, 0.85, 0.54, 0.74, 0.89, 0.52, 0.50, 0.67, 0.65, 0.64, 0.72,
				       1.0, 0.68, 0.84, 0.78])
		pvap = rho * tempK / 217.
		pda = pres - pvap
		den = 3.335e16 * rho
		ti = 300. / tempK
		ti2 = ti**2.5
		
		# continuum terms
		con = (5.43e-10*1.105*pda*ti**3 + 1.8e-8*0.79*pvap*ti**7.5)*pvap*freq**2
		
		# add resonances
		loc_sum = 0.
		nlines = 15
		df = np.empty(2)
		for i in range(nlines):
			width = w3[i]*pda*ti**x[i] + ws[i]*pvap*ti**xs[i]
			wsq = width**2
			s = s1[i]*ti2*np.exp(b2[i]*(1.-ti))
			df[0] = freq - fl[i]
			df[1] = freq + fl[i]
		#  use clough's definition of local line contribution
			base = width/(562500. + wsq)
		#  do for positive and negative resonances
			res = 0.
		#do j=1,2
			for j in range(2):
				if(abs(df[j]) < 750.):
					res = res + width/(df[j]**2+wsq) - base
			
			loc_sum = loc_sum + s*res*(freq/fl[i])**2
		'''
		n_cores = multiprocessing.cpu_count()
		if n_cores > 1:
			n_cores = n_cores - 1
		pool = multiprocessing.Pool(n_cores)
		args = [(w3_s,pda,ti,x_s,ws_s,pvap,xs_s,s1_s,ti2,b2_s,freq,fl_s) for w3_s,x_s,ws_s,xs_s,s1_s,b2_s,fl_s in zip(w3,x,ws,xs,s1,b2,fl)]
		for result in pool.starmap(forLoopH2o,args):
			if result:
				loc_sum = loc_sum + result
		'''
		absh2o = 0.3183e-4*den*loc_sum + con

		return absh2o

def forLoopH2o(w3,pda,ti,x,ws,pvap,xs,s1,ti2,b2,freq,fl):
	df = np.empty(2)
	width = w3*pda*ti**x + ws*pvap*ti**xs
	wsq = width**2
	s = s1*ti2*np.exp(b2*(1.-ti))
	df[0] = freq - fl
	df[1] = freq + fl
	#  use clough's definition of local line contribution
	base = width/(562500. + wsq)
	#  do for positive and negative resonances
	res = 0.
	#do j=1,2
	for j in range(2):
		if(abs(df[j]) < 750.):
			res = res + width/(df[j]**2+wsq) - base
	return s*res*(freq/fl)**2

def getDescriptor():
    
    descriptorFile = np.array([
      #['hydro_name' 'as_ratio' 'liq_ice' 'rho_ms' 'a_ms' 'b_ms' 'alpha_as'
      # 'beta_as' # 'moment_in' 'nbin' 'dist_name' 'p_1' 'p_2' 'p_3' 'p_4' 
      #'d_1' 'd_2' 'scat_name' 'vel_size_mod' 'canting']
      ('cwc_q', -99.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1, 
        'mono', -99.0, -99.0, -99.0, -99.0, 2e-05, -99.0, 'mie-sphere',
        'khvorostyanov01_drops', -99.0)], 
      dtype=[('hydro_name', 'S15'), ('as_ratio', '<f8'), ('liq_ice', '<i8'), 
             ('rho_ms', '<f8'), ('a_ms', '<f8'), ('b_ms', '<f8'), 
             ('alpha_as', '<f8'), ('beta_as', '<f8'), ('moment_in', '<i8'), 
             ('nbin', '<i8'), ('dist_name', 'S15'), ('p_1', '<f8'), 
             ('p_2', '<f8'), ('p_3', '<f8'), ('p_4', '<f8'), ('d_1', '<f8'), 
             ('d_2', '<f8'), ('scat_name', 'S15'), ('vel_size_mod', 'S30'), 
             ('canting', '<f8')])
    
    return descriptorFile 



def getAtmAttPamtra(heightArr,temp,relHum,press,radarFreqs):

	''' 
	use PAMTRA to calculate attenuation due to water vapour, N2 and O2
	taken from tripexProcessing from Jose Dias-Neto
	Parameters
	----------
	heightArr: height array in m
	temp: temperature in Kelvin
	relHum: relative humidity with respect to water
	press: pressure in Pa
	radarFreqs: radar frequencies to use
	Returns
	----------
	attenuation calculated from Pamtra
	'''
    #relHum = atmFunc.speHumi2RelHum(speHum, temp, press)#[%]
    #vaporPress = atmFunc.calcVaporPress(speHum, temp, press)#[Pa] (!! variable name)
    #waterVaporDens = atmFunc.calcVaporDens(vaporPress, temp)#[kg/m^3]
    #dryAirDens = atmFunc.calcDryAirDens(press, waterVaporDens, temp)#[kg/m^3]
    
	descriptorFile = getDescriptor()

	pam = pyPamtra.pyPamtra()
	for hyd in descriptorFile:
		pam.df.addHydrometeor(hyd)

	pam.nmlSet['active'] = True
	pam.nmlSet['passive'] = False
	pam.nmlSet["radar_attenuation"] = 'bottom-up'

	pamData = dict()
	print(heightArr.shape)
	pamData['hgt'] = np.array([heightArr])
	pamData['temp'] = np.array([temp])
	pamData['relhum'] = np.array([relHum])
	pamData['press'] = np.array([press])

	pam.createProfile(**pamData)
	pam.runParallelPamtra(radarFreqs, pp_deltaX=1, pp_deltaY=1, pp_deltaF=1, pp_local_workers=8)

	return pam.r



