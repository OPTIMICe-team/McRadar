#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:13:32 2020

@author: dori
"""

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
plt.close('all')


column_names = ['wl', # [31.557101, 8.565499, 3.155710]
                'elv', # always 90
                'meanAngle', # it is used only for prolate
                'cantingStd', # always 1.0
                'radius', # 8660
                'rho', # 8706
                'as_ratio' # 8769
                ]

data = pd.read_csv('/home/dori/table_McRadar.txt',
                   delim_whitespace=True,
                   header=None,
                   names=column_names) # 26463 lines

wls = data.wl.drop_duplicates()
datalist = [data[data.wl==wl] for wl in wls]
dataX, dataKa, dataW = datalist

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataX.radius, dataX.rho, dataX.as_ratio)
ax.set_xlabel('radius mm')
ax.set_ylabel('density g/cm3')
ax.set_zlabel('aspect ratio')

names = ['time', 'mTot', 'sHeight', 'vel', 'dia', 
             'area', 'sMice', 'sVice', 'sPhi', 'sRhoIce',
             'igf', 'sMult', 'sMrime', 'sVrime']
data_mc = pd.read_csv('/home/dori/develop/McRadar/tests/data_test.dat',
                      header=None, names=names)
data_mc['rho'] = 6.0e-3*data_mc.mTot/(np.pi*data_mc.dia**3*data_mc.sPhi**(-2+3*(data_mc.sPhi<1).astype(int)))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(data_mc.dia*0.5e3, data_mc.rho, data_mc.sPhi)
ax1.set_xlabel('radius mm')
ax1.set_ylabel('density g/cm3')
ax1.set_zlabel('aspect ratio')

for ii in range(0,360,1):
    ax1.view_init(elev=10., azim=ii)
    fig1.savefig("pics/movie{:04d}.png".format(ii))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
def init():
    ax.scatter(data_mc.dia*0.5e3, data_mc.rho, data_mc.sPhi)
    ax.set_xlabel('radius mm')
    ax.set_ylabel('density g/cm3')
    ax.set_zlabel('aspect ratio')
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=3600, interval=20, blit=True)
# Save
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])