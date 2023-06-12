import numpy as np
from scipy import constants
import matplotlib.pyplot as plt 
import pandas as pd
import xarray as xr
import os
import mcradar as mcr
from mcradar import *
from mcradar.tableOperator import creatRadarCols

def Nexp(D, lam):
    return np.exp(-lam*D)

def dB(x):
    return 10.0*np.log10(x)

def Bd(x):
    return 10.0**(0.1*x)
def Modified_gamma(D, lam, mu, nu=1):
	return D**mu*np.exp(-lam*D**nu)


lutPath = '../LUT/DDA/'

frequency =  np.array([9.6e9, 35.6e9, 94.0e9]) # frequencies
Dmax = np.linspace(0.01e-3, 10.0e-3, 100) # list of sizes
lams = 1.0/np.linspace(0.1e-3, 4.0e-3, 2) # list of lambdas
PSD = 30.0*np.array(Nexp(Dmax, 1/5e-3)) 
#PSD = 30.0*np.array(Modified_gamma(Dmax,1/5e-3,0.5))#,3.5)) #N0: number of particles per Dbin

am=0.02522677;bm=2.19978322
mass = am*Dmax**bm
#av=5.97000795;bv=0.45396479
aa=0.07986;ba=1.87763648
area = aa*Dmax**ba
vel = fall_velocity_HW(area,mass,Dmax) 

#- generate table similar to McSnow output containing the masses, Dia,... for which single particle Ze is to be calculated
data = {'dia':Dmax,'mTot':mass,'sPhi':np.ones_like(mass),'sNmono':np.ones_like(mass)*3,'area':area,'vel':-1*vel}
dataTable = pd.DataFrame(data = data).to_xarray()
#- load dicsettings
dicSettings = mcr.loadSettings(PSD=True,#'mass2fr.nc',#inputPath+'mass2fr.nc',
                               elv=np.array([90]), freq=frequency,gridBaseArea=1,maxHeight=100,
                               ndgsVal=50,heightRes=36,convolute=True,#k_theta=0.1,k_phi=0,k_r=0,#shear_height0=700,shear_height1=800,
                               scatSet={'mode':'DDA','lutPath':lutPath})
#- calculate single particle scattering
dataTable = creatRadarCols(dataTable, dicSettings)
Zepart = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
						       			dataTable, ndgs=dicSettings['ndgsVal'],
						        		scatSet=dicSettings['scatSet'])

#Zepart = Zepart/(2*np.pi) # TODO: check when 2pi is needed!! In order to have same scattering as snowScatt I needed to divide by 2pi
# now lets calculate correct Doppler spectrum:
specTable = xr.Dataset()
dataTable = dataTable.sortby('vel')
dataTable['sZePH'] = Zepart.sZeH*PSD*np.gradient(Dmax)/np.gradient(vel) # need to go from size to vel
dataTable['sZePV'] = Zepart.sZeV*PSD*np.gradient(Dmax)/np.gradient(vel) # need to go from size to vel
group = dataTable.groupby_bins('vel', dicSettings['velBins'],labels=dicSettings['velCenterBin']).mean() # get correct Doppler resolution
specTable['spec_H'] = group['sZePH'].rename({'vel_bins':'vel'})
specTable['spec_V'] = group['sZePV'].rename({'vel_bins':'vel'})

#- now lets convolute the spectra!
centerHeight = 100
for wl,th in zip(dicSettings['wl'],dicSettings['theta']/2./180.*np.pi):
	for elv in dicSettings['elv']:
		specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,dicSettings['velCenterBin'],dicSettings['eps_diss'],
					                                      dicSettings['noise_pow'],dicSettings['nave'],th,dicSettings['uwind'],dicSettings['time_int'],centerHeight,0,0,0,dicSettings['tau'])
		specTable['spec_V'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V'].sel(wavelength=wl,elevation=elv).fillna(0),wl,dicSettings['velCenterBin'],dicSettings['eps_diss'],
					                                      dicSettings['noise_pow'],dicSettings['nave'],th,dicSettings['uwind'],dicSettings['time_int'],centerHeight,0,0,0,dicSettings['tau'])
		
print(specTable)
fig,ax=plt.subplots()
for wl,band in zip(dicSettings['wl'],['X','Ka','W']):
	ax.plot(specTable.vel,dB(specTable.spec_H.sel(wavelength=wl,elevation=90)),label=band+'-band')
ax.legend()#fontsize=16)
ax.grid()
ax.set_ylabel('Ze [dBz]',fontsize=18)
ax.set_ylim([-50, 0])
ax.set_xlabel('velocity [m/s]',fontsize=18)
ax.tick_params(axis='both',labelsize=16)
plt.tight_layout()
plt.savefig('PSD_McRadar_one_height_exp.png')
plt.show()






















