import numpy as np
from scipy import constants
import matplotlib
import xarray as xr
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import mcradar as mcr
def getNewNipySpectral():

    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    numEnt = 15

    viridis = cm.get_cmap('nipy_spectral', 256)
    newcolors = viridis(np.linspace(0, 1, 256))

    colorSpace = np.linspace(198, 144, numEnt)/256
    colorTest=np.zeros((numEnt,4))
    colorTest[:,3] = 1
    colorTest[:,0]=colorSpace

    newcolors[- numEnt:, :] = colorTest
    newcmp = ListedColormap(newcolors)

    return newcmp

freq = np.array(9.6e9)
elv = 30
strFreq = '{:.1e}'.format(freq)
print(strFreq)
output = xr.open_dataset('compPol.nc')
print('plotting the spetra')
wls = (constants.c / freq) * 1e3,
output.vel.values = output.vel.values*-1
#print(output.vel.values)
#quit()
for wl in wls:
    print(wl)
    wlStr = '{:.2e}'.format(wl)
    fig,ax=plt.subplots(figsize=(8,7))
    mcr.lin2db(output['spec_H_{0}'.format(wlStr)]).plot(ax=ax,vmin=-30, vmax=10,cmap=getNewNipySpectral(),cbar_kwargs={'label':r'[dB]'})
    plt.title('Ze_H_spec McSnow rad: {0}GHz, elv: {1}'.format(strFreq, str(elv)),fontsize=16)
    ax.set_xlabel(r'vel [ms$^{-1}$]',fontsize=16)
    ax.set_ylabel('range [m]',fontsize=16)
    ax.tick_params(labelsize='large')
    ax.set_ylim(0,5000)
    ax.set_xlim(-3, 0)
    ax.grid(b=True,linestyle='-.')
    plt.savefig('plots/leonie_setup/Leonie_spec_Ze_H_{0}_cyl.png'.format(strFreq), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('plotting the ZDR spetra')
for wl in wls:
    wlStr = '{:.2e}'.format(wl)
    fig,ax=plt.subplots(figsize=(8,7))
    (mcr.lin2db(output['spec_H_{0}'.format(wlStr)])-mcr.lin2db(output['spec_V_{0}'.format(wlStr)])).plot(ax=ax,vmin=-1, vmax=10,cmap=getNewNipySpectral(),cbar_kwargs={'label':r'[dB]'})
    ax.set_title('ZDR McSnow rad: {0}GHz, elv: {1}'.format(strFreq, str(elv)),fontsize=16)
    ax.set_xlabel(r'vel [ms$^{-1}$]',fontsize=16)
    ax.set_ylabel('range [m]',fontsize=16)
    ax.tick_params(labelsize='large')
    ax.set_ylim(0,5000)
    ax.set_xlim(-3, 0)
    ax.grid(b=True,linestyle='-.')
    plt.savefig('plots/leonie_setup/Leonie_spec_ZDR_{0}_cyl.png'.format(strFreq), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('plotting Ze')
for wl in wls:
    wlStr = '{:.2e}'.format(wl)
    fig,ax=plt.subplots(figsize=(8,7))
    mcr.lin2db(output['spec_H_{0}'.format(wlStr)].sum(dim='vel')).plot(ax=ax,y='range',lw=3)
    ax.set_xlabel('Ze_H [dB]',fontsize=16)
    ax.set_ylabel('range [m]',fontsize=16)
    ax.set_title('Ze_H McSnow rad: {0}GHz elv: {1}'.format(strFreq, str(elv)),fontsize=16)
    ax.set_ylim(0, 5000)
    ax.tick_params(labelsize='large')
    #plt.xlim(-3, 0)
    ax.grid(b=True,linestyle='-.')
    plt.savefig('plots/leonie_setup/Leonie_Ze_H_{0}_cyl.png'.format(strFreq), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('plotting ZDR')
for wl in wls:
    wlStr = '{:.2e}'.format(wl)
    fig,ax=plt.subplots(figsize=(8,7))
    (mcr.lin2db(output['spec_H_{0}'.format(wlStr)].sum(dim='vel'))-mcr.lin2db(output['spec_V_{0}'.format(wlStr)].sum(dim='vel'))).plot(ax=ax,y='range',lw=3)
    ax.set_xlabel('ZDR [dB]',fontsize=16)
    ax.set_ylabel('range [m]',fontsize=16)
    ax.set_title('ZDR McSnow rad: {0}GHz elv: {1}'.format(strFreq, str(elv)),fontsize=16)
    ax.set_ylim(0, 5000)
    ax.tick_params(labelsize='large')
    #plt.xlim(-3, 0)
    ax.grid(b=True,linestyle='-.')
    plt.savefig('plots/leonie_setup/Leonie_ZDR_{0}_cyl.png'.format(strFreq), format='png', dpi=200, bbox_inches='tight')
    plt.close()

print('plotting the KDP')
for wl in wls:
    wlStr = '{:.2e}'.format(wl)
    fig,ax=plt.subplots(figsize=(8,7))
    output['kdpInt_{0}'.format(wlStr)].plot(ax=ax,y='range', lw=3)
    ax.set_title('KDP McSnow rad: {0}GHz, elv: {1}'.format(strFreq, str(elv)),fontsize=16)
    ax.set_xlabel(r'KDP [°km$^{-1}$]',fontsize=16)
    ax.set_ylabel('range [m]',fontsize=16)
    ax.tick_params(labelsize='large')
    ax.set_ylim(0, 5000)
    ax.grid(True,linestyle='-.')
    plt.savefig('plots/leonie_setup/Leonie_KDP_{0}_cyl.png'.format(strFreq), format='png', dpi=200, bbox_inches='tight')
    plt.close()

               

