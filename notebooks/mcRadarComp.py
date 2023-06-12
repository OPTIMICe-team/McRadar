import mcradar as mcr
import xarray as xr
import numpy as np
import os 

def getApectRatio(radii):
    # imput radii [mm]
    
    # auer et all 1970 (The Dimension of Ice Crystals in Natural Clouds)
    diameter = 2 * radii *1e3 # calculating the diameter in [mu m]
    h = 2.020 * (diameter)**0.449

    as_ratio = h / diameter
    print(as_ratio) 
    return as_ratio

#reading the data file
dataPath = "/net/broebroe/lvonterz/BIMOD/4Jose" 
fileName = "mass2fr_0300-0600min_avtstep_5.ncdf"
filePath = os.path.join(dataPath, fileName)
data = xr.open_dataset(filePath)

#fake time
time = np.ones_like(data.dim_SP_all_av150)

#calculating the aspec ratio
sPhi = np.ones_like(data.dim_SP_all_av150)*np.nan
sPhi = getApectRatio(data.diam * 1e3)
quit()
sPhi[data.mm.values > 1]=0.6
sPhi[data.mm.values == 1]=0.1


#converting to pandas dataframe
dataTable = data.to_dataframe()
dataTable = dataTable.rename(columns={'m_tot':'mTot', 'height':'sHeight', 
                                      'vt':'vel', 'diam':'dia','xi':'sMult'})
#adding required variables
dataTable['time']=time
dataTable['radii_mm'] = dataTable['dia'] * 1e3 /2.# particle radius in mm 
dataTable['mTot_g'] = dataTable['mTot'] * 1e3 # mass in grams
dataTable['dia_cm'] = dataTable['dia'] * 1e2 # diameter in centimeters
dataTable['sPhi']=sPhi
dataTable = dataTable[(dataTable['sPhi'] >= 0.015)]
# dataTable['sMult']=1 #(it deactivates the multiplicity)

#calculating density
dataTable = mcr.tableOperator.calcRho(dataTable)

#settings
dicSettings = mcr.loadSettings(dataPath='_', freq=np.array([9.6e9]), 
                               #maxHeight=3000, minHeight=2500, 
                               #elv=30,
                               heightRes=150, gridBaseArea=5)

#starting the simulation
output = mcr.fullRadar(dicSettings, dataTable)

#saving the data
output.to_netcdf('comp_smult_leo.nc')
#output.to_netcdf('compPol.nc')
