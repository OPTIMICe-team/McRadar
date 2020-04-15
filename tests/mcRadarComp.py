import mcradar as mcr
import xarray as xr
import numpy as np
import pandas as pd
import os 

from IPython.core.debugger import Tracer ; debug=Tracer() #insert this line somewhere to debug
def getApectRatio(radii):
    # imput radii [mm]
    
    # auer et all 1970 (The Dimension of Ice Crystals in Natural Clouds)
    diameter = 2 * radii *1e3 # calculating the diameter in [mu m]
    h = 2.020 * (diameter)**0.449

    as_ratio = h / diameter
    
    return as_ratio

#reading the data file
dataPath = "data" 
fileName = "mass2fr_0300-0600min_avtstep_5.ncdf"
filePath = os.path.join(dataPath, fileName)
data = xr.open_dataset(filePath)

#fake time
time = np.ones_like(data.dim_SP_all_av150)

#calculating the aspec ratio
sPhi = np.ones_like(data.dim_SP_all_av150)*np.nan
sPhi = getApectRatio(data.diam * 1e3)
sPhi[data.mm.values > 1]=0.6

#converting to pandas dataframe
dataTable = data.to_dataframe()
dataTable = dataTable.rename(columns={'m_tot':'mTot', 'height':'sHeight', 
                                      'vt':'vel', 'diam':'dia','xi':'sMult'})

#settings
dicSettings = mcr.loadSettings(dataPath='_', freq=np.array([9.6e9]), 
                               maxHeight=3000, minHeight=2500, 
                               heightRes=5)
#adding required variables
dataTable['radii'] = dataTable['dia'] / 2.# particle radius in m
dataTable['time']=time

PSD_method="bin" #"bin": count SP and their multiplicity in height and size bins; "1D_KDE": #DOES NOT WORK YET!! 1-dimensional kernel density estimate, "discrete_SP": calculate scattering properties of each SP individually
if PSD_method in ["bin","1D_KDE"]:
    
    #some definitions
    nbins = 100 #number of used bins
    n_heights = 50
    model_top = 3850 #[m] #TODO: read this from output
    minR  =-4   #minimum R considered  (log10-space)
    maxR  = 0   #maximum R considered  (log10-space)
    area_box =  5 #[m2] #TODO: read this from output
    Rgrid=np.logspace(minR,maxR,nbins)
    Rgrid_log=np.linspace(minR,maxR,nbins)
    Rgrid_logdiff=Rgrid_log[1]-Rgrid_log[0]
    heightvec_bound = np.linspace(0,model_top,n_heights)
    #heightvec_bound = np.linspace(2900,3000,5) #TODO: remove (only for debugging)
    Vbox =  area_box*heightvec_bound[1]-heightvec_bound[0] #[m3]

    reducedDataTable = pd.DataFrame()
    for i_height in range(len(heightvec_bound)-1):
        print("calculate h=",heightvec_bound[i_height])
        #initialize as many dataFrame as categories
        #one category must have the same particle properties (mass, velocity) at the same size
        dataBINmono = pd.DataFrame(data={"Rgrid": Rgrid}) #initialize dataFrame
        dataBINagg = pd.DataFrame(data={"Rgrid": Rgrid}) #initialize dataFrame

        #select subset of particles in a given height range
        condition_in_height = np.logical_and(heightvec_bound[i_height]<dataTable["sHeight"],heightvec_bound[i_height+1]>dataTable["sHeight"])

        #select monomers and aggregates
        cond_mono=np.logical_and(dataTable["mm"]==1, condition_in_height) #monomers
        cond_agg=np.logical_and(dataTable["mm"]>1, condition_in_height) #aggregates
        datamono = dataTable[cond_mono]    
        dataagg = dataTable[cond_agg]    
        for key in ["sMult","vel","mTot"]:
            dataBINmono[key] = np.zeros_like(Rgrid)
            dataBINagg[key] = np.zeros_like(Rgrid)
        for i_rad,rad in enumerate(Rgrid[:-1]):
            inbinmono = np.logical_and(Rgrid[i_rad]<datamono["radii"],Rgrid[i_rad+1]>datamono["radii"])
            inbinagg = np.logical_and(Rgrid[i_rad]<dataagg["radii"],Rgrid[i_rad+1]>dataagg["radii"])
            if sum(inbinmono)>0:
                for var_key in ["mTot","vel"]:
                     dataBINmono[var_key][i_rad] = datamono[inbinmono].iloc[0][var_key]  #mass in grams #TODO: calculate m (either by picking a particle from inside the bin or from the m-D relation or build an average of the particle)
            if sum(inbinagg)>0:
                for var_key in ["mTot","vel"]:
                    dataBINagg[var_key][i_rad] = dataagg[inbinagg].iloc[0][var_key]  #mass in grams #TODO: calculate m (either by picking a particle from inside the bin or from the m-D relation or build an average of the particle)
        if PSD_method=="bin":
        
            for i_rad,rad in enumerate(Rgrid[:-1]):
                inbinmono = np.logical_and(Rgrid[i_rad]<datamono["radii"],Rgrid[i_rad+1]>datamono["radii"])
                inbinagg = np.logical_and(Rgrid[i_rad]<dataagg["radii"],Rgrid[i_rad+1]>dataagg["radii"])
                if sum(inbinmono)>0:
                    dataBINmono["sMult"][i_rad] = np.nansum(datamono[inbinmono]["sMult"])
                if sum(inbinagg)>0:
                    dataBINagg["sMult"][i_rad] = np.nansum(dataagg[inbinagg]["sMult"])
                #print(i_rad,dataBINmono["sMult"][i_rad],dataBINagg["sMult"][i_rad],dataBINagg["mTot"][i_rad],dataBINagg["vel"][i_rad])
        elif PSD_method=="1D_KDE": #does not work yet!!
            #calculating number density [#/m3]
            #MONOMERS
            dataBINmono["sMult"] = mcr.tableOperator.kernel_estimate(dataTable["radii"][cond_mono],np.log(Rgrid),weight=dataTable["sMult"][cond_mono],sigma0=0.001)*Rgrid_logdiff #/Vbox #TODO:
            #AGGREGATES
            dataBINagg["sMult"] = mcr.tableOperator.kernel_estimate(dataTable["radii"][cond_agg],np.log(Rgrid),weight=dataTable["sMult"][cond_agg])*Rgrid_logdiff/Vbox 
       
            #for i_rad,Mult in enumerate(dataBINagg["sMult"]): 
            #    print(i_rad,dataBINmono["sMult"][i_rad],dataBINagg["sMult"][i_rad],dataBINagg["mTot"][i_rad],dataBINagg["vel"][i_rad])

        #some general properties and conversions which are independent of the actual SP-list
        dataBINmono['radii_mm'] = dataBINmono['Rgrid'] * 1e3 # particle radius in mm 
        dataBINagg['radii_mm'] = dataBINagg['Rgrid'] * 1e3 # particle radius in mm 
        dataBINmono['sPhi'] = getApectRatio(dataBINmono.radii_mm)
        dataBINagg['sPhi'] = 0.6
        for df in [dataBINmono,dataBINagg]: 
            
            df['dia_cm'] = df['Rgrid'] * 1e2*2 # particle radius in mm 
            df['time']=np.ones_like(df.radii_mm)
            df['sHeight'] = (heightvec_bound[i_height+1]+heightvec_bound[i_height+1])/2
            df['mTot_g'] = dataTable['mTot'] * 1e3 # mass in grams
            #calculating density
            df = mcr.tableOperator.calcRho(df)
            df = df[(df['sPhi'] >= 0.015)] #TODO: this kills everything larger than 3.8mm
            reducedDataTable = pd.concat([reducedDataTable, df])

    reducedDataTable = reducedDataTable[(reducedDataTable['sMult']>1.0)]
    print(reducedDataTable)
    print("?") 
    #starting the simulation
    output = mcr.fullRadar(dicSettings, reducedDataTable)
    print(output)

elif PSD_method=="discrete_SP":
    #adding required variables
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
                                   maxHeight=3000, minHeight=2500, 
                                   heightRes=5)
    #starting the simulation
    output = mcr.fullRadar(dicSettings, dataTable)

    #saving the data
    #output.to_netcdf('comp_smult1.nc')
    output.to_netcdf('comp.nc')
debug()
