import numpy as np
import mcradar as mcr
from mcradar import *
from mcradar.tableOperator import creatRadarCols
from scipy import constants
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd
import xarray as xr
import os

import snowScatt

from snowScatt.instrumentSimulator.radarMoments import Ze
from snowScatt.instrumentSimulator.radarSpectrum import dopplerSpectrum
from snowScatt.instrumentSimulator.radarSpectrum import sizeSpectrum
from snowScatt._compute import backscatVel
from snowScatt.instrumentSimulator.radarMoments import specific_reflectivity
from snowScatt import refractiveIndex
g = 9.81 # gravitational acceleration [m/s^2]
rho_i = 917.6 # density of ice [kg/m^3]
def Nexp(D, lam):
    return np.exp(-lam*D)

def dB(x):
    return 10.0*np.log10(x)

def Bd(x):
    return 10.0**(0.1*x)
def gammadis(D, lam, mu, nu):
	return D**mu*np.exp(-lam*D**nu)

def fall_velocity_HW(area, mass, D_max, T, P):
    """The Heymsfield-Westbrook fall velocity.

    Args:
        area: Projected area [m^2].
        mass: Particle mass [kg].
        D_max: Particle maximum dimension [m].
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The fall velocity [m/s].
    """
    do_i = 8.0
    co_i = 0.35

    rho_air = air_density(T, P)
    eta = air_dynamic_viscosity(T)

    # modified Best number eq. on p. 2478
    Ar = area / (np.pi/4)
    Xbest = rho_air * 8.0 * mass * g * D_max / (eta**2 * np.pi * 
        np.sqrt(Ar))

    # Re-X eq. on p. 2478
    c1 = 4.0 / ( do_i**2 * np.sqrt(co_i) )
    c2 = 0.25 * do_i**2
    bracket = np.sqrt(1.0 + c1*np.sqrt(Xbest)) - 1.0
    Re = c2*bracket**2

    return eta * Re / (rho_air * D_max)
def air_kinematic_viscosity(T, P):
    """The kinematic viscosity of air.

    Args:
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The kinematic viscosity [m^2/s].
    """
    rho = air_density(T, P)
    mu = air_dynamic_viscosity(T)
    return mu/rho


def air_dynamic_viscosity(T):
    """The kinematic viscosity of air.

    Args:
        T: Ambient temperature [K].

    Returns:
        The kinematic viscosity [Pa/s].
    """
    mu0 = 1.716e-5
    T0 = 273.15
    C = 111.0
    return mu0 * ((T0+C)/(T+C)) * (T/T0)**1.5


def air_density(T, P):
    """The density of air.

    Args:
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The kinematic viscosity [Pa/s].
    """
    R = 28704e-2 # gas constant for air
    return P / (T*R)

frequency =  np.array([9.6e9, 35.6e9, 94.0e9]) # frequencies
temperature = 270.0
Nangles = 721

Dmax = np.linspace(0.01e-3, 10.0e-3, 10000) # list of sizes
lams = 1.0/np.linspace(0.1e-3, 4.0e-3, 2) # list of lambdas
am=0.02522677;bm=2.19978322
mass = am*Dmax**bm
#- use snowScatt to calculate spectrum
particle = 'vonTerzi_dendrite'
fig, ax = plt.subplots(figsize=(7,5))
# 
PSD = 10.0*np.stack([np.array(Nexp(Dmax, l)) for l in lams])
#PSD = 10.0*np.array(Nexp(Dmax, 1/5e-3)) 
wl = snowScatt._compute._c/frequency[2]
ls = ['-','--',':']
for i,freq in enumerate(frequency):
	wl = snowScatt._compute._c/freq
	spec0, vel = dopplerSpectrum(Dmax, PSD, wl, particle,
		                         temperature=temperature,mass=mass)
	#spec1 = sizeSpectrum(Dmax, PSD, wl, particle, temperature=temperature)
	#Zx = Ze(Dmax, PSD, wl, particle, temperature=temperature)

	ax.plot(vel, dB(spec0[1].T),C='C0',label='snowScatt {}'.format(freq*1e-9),lw=2,ls=ls[i])
#ax.plot(vel, dB(spectrum.T),label='snowScatt1')


lutPath = '/project/meteo/work/L.Terzi/McRadar/LUT/DDA/'


av=5.97000795;bv=0.45396479
aa=0.07986;ba=1.87763648
area = aa*Dmax**ba
#vel = av*Dmax**bv#fall_velocity_HW(area,mass,Dmax,273.15,1000e2) 
data = {'dia':Dmax,'mTot':mass,'sPhi':np.ones_like(mass),'sNmono':np.ones_like(mass)*3,'area':area,'vel':vel}
dataTable = pd.DataFrame(data = data).to_xarray()
dicSettings = mcr.loadSettings(PSD=True,#'mass2fr.nc',#inputPath+'mass2fr.nc',
                               elv=np.array([90]), freq=frequency,gridBaseArea=1,maxHeight=100,
                               ndgsVal=50,heightRes=36,convolute=True,#k_theta=0.1,k_phi=0,k_r=0,#shear_height0=700,shear_height1=800,
                               scatSet={'mode':'DDA','lutPath':lutPath})


dataTable = creatRadarCols(dataTable, dicSettings)
Zepart = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
						       			dataTable, ndgs=dicSettings['ndgsVal'],
						        		scatSet=dicSettings['scatSet'])
#dopplerVel = dicSettings['velBins']#np.linspace(-10.0, 10.0, 1024)

#velidx = dataTable.vel.argsort()

#dopplerVel = dataTable.vel[velidx]
#print(dopplerVel)
#quit()
#spectrum[: ,velidx], dopplerVel
#for i,P in enumerate(PSD):
P = PSD[1]
Zepart = Zepart/(2*np.pi)
dataTable['sZeP'] = Zepart.sZeH*P*np.gradient(Dmax)/np.gradient(vel)

group = dataTable.groupby_bins('vel', dicSettings['velBins'],labels=dicSettings['velCenterBin']).mean()
spec_H_m = group['sZeP'].rename({'vel_bins':'vel'})

dv = np.abs(np.diff(vel)[0])
dataTable['sZePSD'] = dataTable['sZeP']/dv
group = dataTable.groupby_bins('vel', dicSettings['velBins'],labels=dicSettings['velCenterBin']).sum()
spec_H = group['sZeP'].rename({'vel_bins':'vel'})
spec_H = spec_H*np.diff(spec_H.vel.values)[0]
sZe = dataTable.assign_coords({'index':dataTable.vel}).drop('vel').rename({'index':'vel'})
sZe = sZe.reindex({'vel':dicSettings['velCenterBin']},method='nearest',tolerance=np.diff(dicSettings['velCenterBin'])[0])
sZeH = sZe.sZePSD*np.diff(sZe.vel.values)[0]
#print(sZe)
#quit()
#sel(wavelength=spec_H.wavelength))
#spec_H = dataTable['sZe']
#print(spec_H)
#quit()
#ax.plot(sZeH.vel,dB(sZeH.sel(wavelength=sZeH.wavelength[0],elevation=sZeH.elevation[0])),label='McRadar')
for i,wl in enumerate(dataTable.wavelength):
	#ax.plot(vel,dB(dataTable['sZeP'].sel(wavelength=wl,elevation=dataTable.elevation[0])),label='McRadar not regridded',C='C1',ls=ls[i],lw=2)
	#ax.plot(spec_H.vel,dB(spec_H.sel(wavelength=spec_H.wavelength[0],elevation=spec_H.elevation[0])),label='McRadar_sum')
	ax.plot(spec_H_m.vel,dB(spec_H_m.sel(wavelength=wl,elevation=spec_H_m.elevation[0])),label='McRadar',C='C1',lw=2,ls=ls[i])
#plt.plot(dataTable.vel,dB(dataTable.sZePSD.sel(wavelength=dataTable.wavelength[0],elevation=dataTable.elevation[0])),ls='--')

ax.legend()#fontsize=16)
ax.grid()
ax.set_ylabel('Ze [dBz]',fontsize=18)
ax.set_ylim([-50, 0])
ax.set_xlabel('velocity [m/s]',fontsize=18)
ax.tick_params(axis='both',labelsize=16)
plt.tight_layout()
plt.savefig('test_PSD_McRadar_all.png')
plt.show()






















