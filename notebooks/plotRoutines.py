'''
This is the beginning of plotting routines for McSnow output
'''
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import mcradar as mcr
import numpy as np

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

def plotArSpec(dicSettings,mcTable,velBins,inputPath):
    for i, heightEdge0 in enumerate(dicSettings['heightRange']):
        heightEdge1 = heightEdge0 + dicSettings['heightRes']
        height = heightEdge0+dicSettings['heightRes']/2
        mcTableTmp = mcTable[(mcTable['sHeight']>heightEdge0) &
                             (mcTable['sHeight']<=heightEdge1)].copy()
        binVel,sVel = pd.cut(mcTableTmp['vel'],bins=velBins,retbins=True)
        binnedPhi = mcTableTmp.groupby(binVel)[['sPhi']].mean()
        if i == 0:
            binnedXR = xr.DataArray(binnedPhi.values,
                                    dims=('vel','height'),
                                    coords={'height':height.reshape(1),'vel':velBins[0:-1]})
        else:
            tmpXR = xr.DataArray(binnedPhi,
                                    dims=('vel','height'),
                                    coords={'height':height.reshape(1),'vel':velBins[0:-1]})
            binnedXR = xr.concat([binnedXR,tmpXR],dim='height')
    binnedXR.plot(x='vel',y='height',vmin=0.1,vmax=5,cmap=getNewNipySpectral(),cbar_kwargs={'label':'ar'})
    plt.grid(True,ls='-.')
    plt.xlabel('vel [m/s]')
    plt.savefig(inputPath+'1d_habit_spec_ar.png')
    plt.close()

def plotMoments(dicSettings,output,inputPath):
    for wl in dicSettings['wl']:

        wlStr = '{:.2e}'.format(wl)
        #plot KDP    
        fig,axes = plt.subplots(ncols=3,figsize=(15,5),sharey=True)
        output['kdpInt_{0}'.format(wlStr)].plot(ax=axes[0],y='range', lw=2)
        axes[0].set_title('rad: {0} elv: {1}, KDP'.format(wlStr, dicSettings['elv']))
        #axes[0].set_ylim(0, 5000)
        axes[0].grid(True,ls='-.')
        #plt.savefig(inputPath+'1d_habit_KDP_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
        #plt.close()
     
        # plot ZDR
        axes[1].plot(mcr.lin2db(output['Ze_H_{0}'.format(wlStr)])-mcr.lin2db(output['Ze_V_{0}'.format(wlStr)]),output['range'],linewidth=2)
        axes[1].set_xlabel('ZDR [dB]')
        axes[1].set_title('ZDR')
        #plt.xlim(-3, 0)
        axes[1].grid(b=True,ls='-.')
        #plt.savefig(inputPath+'1d_habit_ZDR_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
        #plt.close()

        axes[2].plot(mcr.lin2db(output['Ze_H_{0}'.format(wlStr)]),output['range'],linewidth=2)
        axes[2].set_xlabel('Z_H [dB]')
        axes[2].set_title('Ze_H')
        #plt.ylim(0, 5000)
        #plt.xlim(-3, 0)
        axes[2].grid(b=True,ls='-.')
        plt.tight_layout()
        plt.savefig(inputPath+'1d_habit_moments_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
        plt.close()

def plotSpectra(dicSettings,output,inputPath):
    for wl in dicSettings['wl']:

        wlStr = '{:.2e}'.format(wl)
        fig,axes = plt.subplots(ncols=2,figsize=(10,5),sharey=True)
        mcr.lin2db(output['spec_H_{0}'.format(wlStr)]).plot(ax=axes[0],vmin=-30, vmax=5, cmap=getNewNipySpectral(),cbar_kwargs={'label':'sZe [dB]'})
        axes[0].set_title('Ze_H_spec rad: {0} elv: {1}'.format(wlStr, dicSettings['elv']))
        #plt.ylim(0,5000)
        axes[0].set_xlim(-3, 0)
        axes[0].grid(True,ls='-.')

        (mcr.lin2db(output['spec_H_{0}'.format(wlStr)])-mcr.lin2db(output['spec_V_{0}'.format(wlStr)])).plot(ax=axes[1],vmin=-3, vmax=1,
                                                                                                             cmap=getNewNipySpectral(),
                                                                                                             cbar_kwargs={'label':'sZDR [dB]'})
        axes[1].set_title('ZDR rad: {0} elv: {1}'.format(wlStr, dicSettings['elv']))
        axes[1].set_xlim(-3, 0)
        axes[1].grid(True,ls='-.')
        plt.tight_layout()
        plt.savefig(inputPath+'1d_habit_spectra_{0}.png'.format(wlStr), format='png', dpi=200, bbox_inches='tight')
        plt.close()


