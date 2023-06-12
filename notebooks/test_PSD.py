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
g = 9.81 # gravitational acceleration [m/s^2]
rho_i = 917.6 # density of ice [kg/m^3]
def dB(x):
    return 10.0*np.log10(x)
def Nexp(D, lam):
	return np.exp(-lam*D)
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

lutPath = '/project/meteo/work/L.Terzi/McRadar/LUT/DDA/'

Dmax = np.linspace(0.01e-3, 5.0e-3, 2000) # list of sizes
#-calculate aspect ratios according to Leinonen 2015 for plates:
a = Dmax/2
L = 1.737e-3*a**0.474
#mD in g,cm: m = 3.76*10**-2*d**3.31
am = 3.76e-2; bm = 3.31
mass = 100**bm*am/1000*Dmax**bm
ar = L/Dmax
mono_gam_plates = 2.00                            
mono_sig_plates = 0.6495
area = mono_sig_plates*Dmax**mono_gam_plates
vel = fall_velocity_HW(area,mass,Dmax,273.15,1000e2) 
data = {'dia':Dmax,'mTot':mass,'sPhi':ar,'sNmono':np.ones_like(mass),'area':area,'vel':-1*vel}
dataTable = pd.DataFrame(data = data).to_xarray()

#lams = 1.0/np.linspace(0.1e-3, 4.0e-3, 10) # list of lambdas
#PSDs = 10.0*np.stack([np.array(Nexp(Dmax, l)) for l in lams])
PSD = 10.0*np.array(Nexp(Dmax, 1/5e-3)) 

dicSettings = mcr.loadSettings(PSD=True,#'mass2fr.nc',#inputPath+'mass2fr.nc',
                               elv=np.array([90]), freq=np.array([9.6e9]),gridBaseArea=1,maxHeight=100,
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

dataTable['sZePSD'] = Zepart.sZeH*PSD*np.gradient(Dmax)/np.gradient(vel)
group = dataTable.groupby_bins('vel', dicSettings['velBins'],labels=dicSettings['velCenterBin']).sum()
spec_H = group['sZePSD'].rename({'vel_bins':'vel'})
spec_H = spec_H*np.diff(spec_H.vel.values)[0]
sZe = dataTable.assign_coords({'index':dataTable.vel}).drop('vel').rename({'index':'vel'})
sZe = sZe.reindex({'vel':dicSettings['velCenterBin']},method='nearest',tolerance=np.diff(dicSettings['velCenterBin'])[0])
sZeH = sZe.sZePSD#*np.diff(sZe.vel.values)[0]
#print(sZe)
#quit()
#sel(wavelength=spec_H.wavelength))
#spec_H = dataTable['sZe']
#print(spec_H)
#quit()
plt.plot(sZeH.vel,dB(sZeH.sel(wavelength=sZeH.wavelength[0],elevation=sZeH.elevation[0])))
#plt.plot(spec_H.vel,dB(spec_H.sel(wavelength=spec_H.wavelength[0],elevation=spec_H.elevation[0])))
plt.plot(dataTable.vel,dB(dataTable.sZePSD.sel(wavelength=dataTable.wavelength[0],elevation=dataTable.elevation[0])),ls='--')
#plt.plot(-1*vel,dB(sZe.sel(wavelength=sZe.wavelength[0],elevation=sZe.elevation[0])))
plt.show()
#print(sZe) 
#z*psd*np.gradient(diameters)/np.gradient(vel)

#plt.show()























