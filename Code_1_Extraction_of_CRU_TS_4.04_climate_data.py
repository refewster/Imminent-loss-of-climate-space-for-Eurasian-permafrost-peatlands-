# -*- coding: cp1252 -*-
########################################################################################################
########################################################################################################
"""
                                                       Code_1_Extraction_of_CRU_TS_4.04_climate_data
"""
########################################################################################################
########################################################################################################
"""
Title: Code_1_Extraction_of_CRU_TS_4.04_climate_data
Author: Richard Fewster (gy15ref@leeds.ac.uk)
Reference: Fewster, R.E., Morris, P.J., Ivanovic, R.F., Swindles, G.T., Peregon, A., and Smith, C. Imminent loss of climate space for Eurasian permafrost peatlands. (in press).
Description: This script reads in CRU observational climate netcdf files, slices them to time periods of interest, calculates monthly means, and then
outputs the data in csv and netcdf format.
Most Recent Update: 18/05/2021

"""
########################################################################################################
########################################################################################################
########################################################################################################
"""
STEP 1: IMPORT LIBRARIES AND DATA FILES
"""
########################################################################################################
"""
(1.1) Import required libraries.
"""
print('(1.1) Importing libraries...')

import xarray as xr
import datetime as dt
import numpy as np
import pandas as pd
import re
from netCDF4 import Dataset, date2index, num2date, date2num
import scipy
import gridfill
import iris

print('Import complete')

print('Import observational datasets...')

# Load in observational temperature dataset
CRU_tmp_file = r"G:\Climate_Data\3_Observational_data\CRU data\CRU_TS_404\cru_ts4.04.1901.2019.tmp.dat.nc"
CRU_tmp_dset = xr.open_mfdataset(CRU_tmp_file, combine='by_coords')

# Load in observational precipitation dataset
CRU_pre_file =r"G:\Climate_Data\3_Observational_data\CRU data\CRU_TS_404\cru_ts4.04.1901.2019.pre.dat.nc"
CRU_pre_dset = xr.open_mfdataset(CRU_pre_file, combine='by_coords')

print('Setup export directory...')

# Export path for temperature files
tmp_DIR = r'G:\Climate_Data\3_Observational_data\CRU data\CRU_TS_404\Outputs\CRU_1961_1990_tmp_data_'

# Export path for precipitation files
pre_DIR = r'G:\Climate_Data\3_Observational_data\CRU data\CRU_TS_404\Outputs\CRU_1961_1990_pre_data_'

print('Step 1: Import and Setup complete')

########################################################################################################
"""
STEP 2: SUBSET DATA FILES AND CALCULATE MONTHLY AVERAGES
"""
########################################################################################################
print('Subsetting data files and calculating monthly averages...')
#Temperature
CRU_tmp_slice_xr = CRU_tmp_dset.sel(time=slice("1961-01-16", "1990-12-16"))
CRU_tmp_xr = CRU_tmp_slice_xr['tmp'].groupby("time.month").mean('time',keep_attrs=True)

# Precipitation
CRU_pre_slice_xr = CRU_pre_dset.sel(time=slice("1961-01-16", "1990-12-16"))
CRU_pre_xr = CRU_pre_slice_xr['pre'].groupby("time.month").mean('time',keep_attrs=True)

print("Step 2: Subsetting data complete")

########################################################################################################
"""
STEP 3: (OPTIONAL) CROP TO REGION OF INTEREST
"""
########################################################################################################

"""
(3.1) Subset to the region of interest
"""
# Select only grid cells within the latitudinal bands:
answer = input('(OPTIONAL) Crop output to study region?:')
if answer.lower().startswith("y"):
      # Temperature
      CRU_tmp_xr = CRU_tmp_xr.sel(lat=slice(44., 90.))

      # Preciptiation
      CRU_pre_xr = CRU_pre_xr.sel(lat=slice(44., 90.))
      
      tmp_DIR= tmp_DIR+'sliced'
      pre_DIR= pre_DIR+'sliced'
      
elif answer.lower().startswith("n"):
    
      tmp_DIR= tmp_DIR+'global'
      pre_DIR= pre_DIR+'global'

else:
        print("Enter either yes/no")

print("Step 3 complete")

########################################################################################################
"""
STEP 4: OUTPUT RESULTS 
"""
########################################################################################################
"""
(4.1) Exporting the results to netcdf format
"""

import sys
# Optional choice to export as netcdf or pass
answer = input('(OPTIONAL) Export data files to netCDF?:')
if answer.lower().startswith("y"):
      print("(4.1) Data export to NetCDF...")
      # Prevent warnings from flashing up - turn off/on as desired
      # Turned off as no issue with 'true divide' (dividing by NaN).
      np.warnings.filterwarnings('ignore')
      # Temperature files
      CRU_tmp_xr.to_netcdf(tmp_DIR+'.nc')
      print('tmp_CRU_1961_1990.nc complete')
      # Precipitation files
      CRU_pre_xr.to_netcdf(pre_DIR+'.nc')
      print('pre_CRU_1961_1990.nc complete')
      # Turn warnings back on
      np.warnings.filterwarnings('default')
elif answer.lower().startswith("n"):
    pass
else:
        print("Enter either yes/no")

"""
(4.2) Export the results as .csv
"""
# Optional choice to export as netcdf or pass
answer = input('(OPTIONAL) Export data files to .csv?:')
if answer.lower().startswith("y"):
      print("(4.2) Data export to .csv...")
# Prevent warnings from flashing up - turn off/on as desired
      np.warnings.filterwarnings('ignore')
# Temperature 
      CRU_hist_tmp = CRU_tmp_xr.rename('mean monthly near-surface temperature (degrees Celsius)') # ensure dataset is named
      hist_tmp_jan = CRU_hist_tmp.sel(month=1) # select tas data from the first month
      hist_tmp_jan_df= hist_tmp_jan.to_dataframe() # turn this data into a pandas dataframe
      hist_tmp_jan_df = hist_tmp_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jan_MMT"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      hist_tmp_feb = CRU_hist_tmp.sel(month=2)
      hist_tmp_feb_df= hist_tmp_feb.to_dataframe()
      hist_tmp_feb_df = hist_tmp_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Feb_MMT"})
      print('##')      
      hist_tmp_mar = CRU_hist_tmp.sel(month=3)
      hist_tmp_mar_df= hist_tmp_mar.to_dataframe()
      hist_tmp_mar_df = hist_tmp_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Mar_MMT"})
      print('###')      
      hist_tmp_apr = CRU_hist_tmp.sel(month=4)
      hist_tmp_apr_df= hist_tmp_apr.to_dataframe()
      hist_tmp_apr_df = hist_tmp_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Apr_MMT"})
      print('####')      
      hist_tmp_may = CRU_hist_tmp.sel(month=5)
      hist_tmp_may_df= hist_tmp_may.to_dataframe()
      hist_tmp_may_df = hist_tmp_may_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "May_MMT"})
      print('#####')      
      hist_tmp_jun = CRU_hist_tmp.sel(month=6)
      hist_tmp_jun_df= hist_tmp_jun.to_dataframe()
      hist_tmp_jun_df = hist_tmp_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jun_MMT"})
      print('######')      
      hist_tmp_jul = CRU_hist_tmp.sel(month=7)
      hist_tmp_jul_df= hist_tmp_jul.to_dataframe()
      hist_tmp_jul_df = hist_tmp_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jul_MMT"})
      print('#######')      
      hist_tmp_aug = CRU_hist_tmp.sel(month=8)
      hist_tmp_aug_df= hist_tmp_aug.to_dataframe()
      hist_tmp_aug_df = hist_tmp_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Aug_MMT"})
      print('########')      
      hist_tmp_sep = CRU_hist_tmp.sel(month=9)
      hist_tmp_sep_df= hist_tmp_sep.to_dataframe()
      hist_tmp_sep_df = hist_tmp_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Sep_MMT"})
      print('#########')      
      hist_tmp_oct = CRU_hist_tmp.sel(month=10)
      hist_tmp_oct_df= hist_tmp_oct.to_dataframe()
      hist_tmp_oct_df = hist_tmp_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Oct_MMT"})
      print('##########')      
      hist_tmp_nov = CRU_hist_tmp.sel(month=11)
      hist_tmp_nov_df= hist_tmp_nov.to_dataframe()
      hist_tmp_nov_df = hist_tmp_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Nov_MMT"})
      print('###########')      
      hist_tmp_dec = CRU_hist_tmp.sel(month=12)
      hist_tmp_dec_df= hist_tmp_dec.to_dataframe()
      hist_tmp_dec_df = hist_tmp_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Dec_MMT"})
      print('############')      
      hist_tmp_df = pd.concat([hist_tmp_jan_df, hist_tmp_feb_df, hist_tmp_mar_df, hist_tmp_apr_df, hist_tmp_may_df, hist_tmp_jun_df, hist_tmp_jul_df, hist_tmp_aug_df, hist_tmp_sep_df, hist_tmp_oct_df, hist_tmp_nov_df, hist_tmp_dec_df], axis=1) # add each variable as a column
      hist_tmp_df = hist_tmp_df.reset_index() # add id column
      hist_tmp_df.index = hist_tmp_df.index + 1 # start id index at 1, not 0
      hist_tmp_df.to_csv(tmp_DIR+'.csv')
      print('CRU_1961_1990_tmp.csv complete')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
# Precipitation files
# Historical 
      CRU_hist_pre = CRU_pre_xr.rename('mean monthly precipitation (mm)') # ensure dataset is named
      hist_pre_jan = CRU_hist_pre.sel(month=1) # select pre data from the first month
      hist_pre_jan_df= hist_pre_jan.to_dataframe() # turn this data into a pandas dataframe
      hist_pre_jan_df = hist_pre_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jan_pre"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      hist_pre_feb = CRU_hist_pre.sel(month=2)
      hist_pre_feb_df= hist_pre_feb.to_dataframe()
      hist_pre_feb_df = hist_pre_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Feb_pre"})
      print('##')
      hist_pre_mar = CRU_hist_pre.sel(month=3)
      hist_pre_mar_df= hist_pre_mar.to_dataframe()
      hist_pre_mar_df = hist_pre_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Mar_pre"})
      print('###')
      hist_pre_apr = CRU_hist_pre.sel(month=4)
      hist_pre_apr_df= hist_pre_apr.to_dataframe()
      hist_pre_apr_df = hist_pre_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Apr_pre"})
      print('####')
      hist_pre_may = CRU_hist_pre.sel(month=5)
      hist_pre_may_df= hist_pre_may.to_dataframe()
      hist_pre_may_df = hist_pre_may_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "May_pre"})
      print('#####')
      hist_pre_jun = CRU_hist_pre.sel(month=6)
      hist_pre_jun_df= hist_pre_jun.to_dataframe()
      hist_pre_jun_df = hist_pre_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jun_pre"})
      print('######')
      hist_pre_jul = CRU_hist_pre.sel(month=7)
      hist_pre_jul_df= hist_pre_jul.to_dataframe()
      hist_pre_jul_df = hist_pre_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jul_pre"})
      print('#######')
      hist_pre_aug = CRU_hist_pre.sel(month=8)
      hist_pre_aug_df= hist_pre_aug.to_dataframe()
      hist_pre_aug_df = hist_pre_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Aug_pre"})
      print('########')
      hist_pre_sep = CRU_hist_pre.sel(month=9)
      hist_pre_sep_df= hist_pre_sep.to_dataframe()
      hist_pre_sep_df = hist_pre_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Sep_pre"})
      print('#########')
      hist_pre_oct = CRU_hist_pre.sel(month=10)
      hist_pre_oct_df= hist_pre_oct.to_dataframe()
      hist_pre_oct_df = hist_pre_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Oct_pre"})
      print('##########')
      hist_pre_nov = CRU_hist_pre.sel(month=11)
      hist_pre_nov_df= hist_pre_nov.to_dataframe()
      hist_pre_nov_df = hist_pre_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Nov_pre"})
      print('###########')
      hist_pre_dec = CRU_hist_pre.sel(month=12)
      hist_pre_dec_df= hist_pre_dec.to_dataframe()
      hist_pre_dec_df = hist_pre_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Dec_pre"})
      print('############')
      hist_pre_df = pd.concat([hist_pre_jan_df, hist_pre_feb_df, hist_pre_mar_df, hist_pre_apr_df, hist_pre_may_df, hist_pre_jun_df, hist_pre_jul_df, hist_pre_aug_df, hist_pre_sep_df, hist_pre_oct_df, hist_pre_nov_df, hist_pre_dec_df], axis=1) # add each variable as a column
      hist_pre_df = hist_pre_df.reset_index() # add id column
      hist_pre_df.index = hist_pre_df.index + 1 # start id index at 1, not 0
      hist_pre_df.to_csv(pre_DIR+'.csv')
      print('CRU_1961_1990_pre.csv complete')
       # Turn warnings back on
      np.warnings.filterwarnings('default')
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
     
elif answer.lower().startswith("n"):
    pass
else:
        print("Enter either yes/no")

print('Step 4: Data Export complete')
################################################################################################################################
print('End.')
sys.exit()
