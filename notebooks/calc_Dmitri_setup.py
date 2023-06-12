# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: Leonie von Terzi


# this calculates the polarimetric variables as in Moisseev et al. 2015 in order to see if I can reproduce his calculations

import numpy as np
import mcradar as mcr
from scipy import constants
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xarray as xr
#mpl.style.use('seaborn')
from pytmatrix.tmatrix import Scatterer
from pytmatrix import refractive, tmatrix_aux, radar, psd, orientation

wlC = tmatrix_aux.wl_C
wlW = tmatrix_aux.wl_W
print(wlC)
print(wlW)
print(wlC/wlW)
quit()

rho_ice=0.917 #-- density of solid ice in g/cm3
am = 6.12e-4 #21.1e-4 #6.12e-4
bm = 2.29 #2.53#2.29
av = 55 #72 #55
bv = 0.48 #0.33#0.48
ah = 9e-3 # 10.7e-3#9e-3
bh = 0.377 #0.431 #0.377
D_vec = np.linspace(0.06,0.5) #-- define Diameters in cm
#D_vec = np.linspace(0.003241213102136767*1e2,0.01671510273827263*1e2)
mass_vec = am*D_vec**bm  #-- define mass in g
vel_vec = av*D_vec**bv  #-- define vel in cm/s
ar_vec = np.asarray([0.01,0.05,0.1])
#ar_vec = np.asarray([1,2,3,4,5,6])
# comment scatterer.or_pdf and scatterer.orient if you are not plotting ar larger 1
scatterer = Scatterer(wavelength=wl)
scatterer.radius_type = Scatterer.RADIUS_MAXIMUM
scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)

#scatterer.or_pdf = orientation.gaussian_pdf(std=1, mean=90)  
#         scatterer.orient = orientation.orient_averaged_adaptive
#scatterer.orient = orientation.orient_averaged_fixed

scatterer.ndgs = 60
#scatterer.ddelt = 1e-5
scatterer.thet0 = 90. - 30.
scatterer.phi0 = 0.
scatterer.thet = scatterer.thet0
scatterer.phi = (scatterer.phi0) % 360.
for ar in ar_vec:
    vol = 4/3*np.pi*(D_vec/2)**3*ar
    rho = mass_vec/vol

    sMat = np.ones_like(D_vec)*np.nan
    Kdp =  np.ones_like(D_vec)*np.nan
    reflect_h = np.ones_like(D_vec)*np.nan
    reflect_v = np.ones_like(D_vec)*np.nan
    Zdr =  np.ones_like(D_vec)*np.nan
    for i, d in enumerate(D_vec):
        print('D',d); print('ar',ar)
        scatterer.axis_ratio = 1/ar
        scatterer.radius = d*1e1/2
        scatterer.m = refractive.mi(wl, rho[i])
        Kdp[i]=radar.Kdp(scatterer)
        Zdr[i] = radar.Zdr(scatterer)
    np.savetxt('/work/lvonterz/pol-scatt/DDA_compute_broe/Tmatrix_X_kdp_ar'+str(ar)+'.txt',np.vstack((D_vec*1e-2,Kdp)).T,fmt="%.6e")
    #plt.plot(D_vec,Kdp,label='ar:%5.3f'%float(ar))
print('loop done')
quit()

#plt.ylim(,9e-5)
plt.grid(True)
plt.title('wl: '+str(wl)+'mm, P1e')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.xlabel('D [cm]')
plt.ylabel(r'KDP [°/km]')
plt.tight_layout()
plt.savefig('KDP_X_test_comp_DDA.png')
plt.show()
