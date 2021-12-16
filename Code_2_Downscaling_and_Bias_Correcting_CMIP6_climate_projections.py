# -*- coding: cp1252 -*-
########################################################################################################
########################################################################################################
"""
                                        Code_2_Downscaling_and_Bias_Correcting_CMIP6_climate_projections
"""
########################################################################################################
########################################################################################################

"""

Title: Code_2_Downscaling_and_Bias_Correcting_CMIP6_climate_projections
Author: Richard Fewster (gy15ref@leeds.ac.uk)
Reference: Fewster, R.E., Morris, P.J., Ivanovic, R.F., Swindles, G.T., Peregon, A., and Smith, C. Imminent loss of climate space for permafrost peatlands in Europe and Western Siberia. (in review).
Description: This program reads in CMIP6 climate netcdf files, slices them to time periods of interest, calculates monthly means, and then
extrapolates, downscales and bias-corrects the outputs following the method of Morris et al. (2018). Functions are included for the export
of this data to csv and netcdf format. This script includes example pathfiles for the processing of ACCESS-CM2 projections.
References:
* Morris, P.J., Swindles, G.T., Valdes, P.J., Ivanovic, R.F., Gregoire, L.J., Smith, M.W., Tarasov, L., Haywood, A.M. and Bacon, K.L., 2018. Global peatland initiation driven by regionally asynchronous warming. Proceedings of the National Academy of Sciences, 115(19), pp.4851-4856.
* Dawson, A. 2018. gridfill. [Online]. [Accessed June 2020]. Available from: https://github.com/ajdawson/gridfill
Most recent update: 29/01/2021

"""


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

"""
(1.2) Import CMIP netCDF files 
"""
print('(1.2) Importing CMIP data files...')

# Import CMIP land-sea mask file
mask_file = r"...\ACCESS-CM2\land\sftlf_fx_ACCESS-CM2_historical_r1i1p1f1_gn.nc"

# Import temperature files (e.g. for ACCESS-CM2...)
tas_file_hist = r"...\ACCESS-CM2\tmp\historical\*.nc"
tas_file_hist = xr.open_mfdataset(tas_file_hist, combine='by_coords')
tas_file_ssp1 = r"...\ACCESS-CM2\tmp\ssp1_26\*.nc"
tas_file_ssp1 = xr.open_mfdataset(tas_file_ssp1, combine='by_coords')
tas_file_ssp2 = r"...\ACCESS-CM2\tmp\ssp2_45\*.nc"
tas_file_ssp2 = xr.open_mfdataset(tas_file_ssp2, combine='by_coords')
tas_file_ssp3 = r"...\ACCESS-CM2\tmp\ssp3_70\*.nc"
tas_file_ssp3 = xr.open_mfdataset(tas_file_ssp3, combine='by_coords')
tas_file_ssp5 = r"...\ACCESS-CM2\tmp\ssp5_85\*.nc"
tas_file_ssp5 = xr.open_mfdataset(tas_file_ssp5, combine='by_coords')

# Import precipitation files (e.g. for ACCESS-CM2...)
pre_file_hist = r"...\ACCESS-CM2\pre\historical\*.nc"
pre_file_hist = xr.open_mfdataset(pre_file_hist, combine='by_coords')
pre_file_ssp1 = r"...\ACCESS-CM2\pre\ssp1_26\*.nc"
pre_file_ssp1 = xr.open_mfdataset(pre_file_ssp1, combine='by_coords')
pre_file_ssp2 = r"...\ACCESS-CM2\pre\ssp2_45\*.nc"
pre_file_ssp2 = xr.open_mfdataset(pre_file_ssp2, combine='by_coords')
pre_file_ssp3 = r"...\ACCESS-CM2\pre\ssp3_70\*.nc"
pre_file_ssp3 = xr.open_mfdataset(pre_file_ssp3, combine='by_coords')
pre_file_ssp5 = r"...\ACCESS-CM2\pre\ssp5_85\*.nc"
pre_file_ssp5 = xr.open_mfdataset(pre_file_ssp5, combine='by_coords')

"""
(1.3) Import observational netCDF files 
"""
print('(1.3) Importing observational data files...')

# Import observational temperature dataset (e.g. for CRU TS 4.02...)
CRU_tmp_file = r"...\cru_ts4.04.1901.2019.tmp.dat.nc"
CRU_tmp_dset = xr.open_mfdataset(CRU_tmp_file, combine='by_coords')

# Import observational precipitation dataset (e.g. for CRU TS 4.02...)
CRU_pre_file =r"...\cru_ts4.04.1901.2019.pre.dat.nc"
CRU_pre_dset = xr.open_mfdataset(CRU_pre_file, combine='by_coords')

"""
(1.4) Setup export directory.
"""
print('(1.4) Setting up export directory...')

# Export path for temperature files (e.g. for ACCESS-CM2...)
tmp_DIR = r'...\ACCESS-CM2_1x_downscaled_monthly_tmp_'

# Export path for precipitation files (e.g. for ACCESS-CM2...)
pre_DIR = r'...\ACCESS-CM2_1x_downscaled_monthly_pre_'

print('Step 1: Import and Setup complete')



########################################################################################################
"""
STEP 2: PROCESS CLIMATE DATA AND CALCULATE AVERAGES
"""
########################################################################################################

"""
(2.1) Slice CMIP data to desired time period and study area.
"""
print('(2.1) Slicing data files to chosen time period...')

# Temperature files sliced to desired time period
tas_hist_slice = tas_file_hist.sel(time=slice("1961-01-16", "1990-12-16"))
tas_ssp1_slice = tas_file_ssp1.sel(time=slice("2090-01-16", "2099-12-16")) 
tas_ssp2_slice = tas_file_ssp2.sel(time=slice("2090-01-16", "2099-12-16"))
tas_ssp3_slice = tas_file_ssp3.sel(time=slice("2090-01-16", "2099-12-16"))
tas_ssp5_slice = tas_file_ssp5.sel(time=slice("2090-01-16", "2099-12-16"))

# Precipitation files sliced to desired time period
pre_hist_slice = pre_file_hist.sel(time=slice("1961-01-16", "1990-12-16"))
pre_ssp1_slice = pre_file_ssp1.sel(time=slice("2090-01-16", "2099-12-16")) 
pre_ssp2_slice = pre_file_ssp2.sel(time=slice("2090-01-16", "2099-12-16"))
pre_ssp3_slice = pre_file_ssp3.sel(time=slice("2090-01-16", "2099-12-16"))
pre_ssp5_slice = pre_file_ssp5.sel(time=slice("2090-01-16", "2099-12-16"))

"""
(2.2) Convert climate data into desired units and average climate values for study time period
"""
print('(2.2) Calculating monthly temperature and precipitation values...')

# Temperature files
# GroupBy subdivides dataset into months before averaging. This code produces mean monthly temperatures.
tas_hist_mean_monthly_K = tas_hist_slice['tas'].groupby("time.month").mean('time',keep_attrs=True)
tas_ssp1_mean_monthly_K = tas_ssp1_slice['tas'].groupby("time.month").mean('time',keep_attrs=True)
tas_ssp2_mean_monthly_K = tas_ssp2_slice['tas'].groupby("time.month").mean('time',keep_attrs=True)
tas_ssp3_mean_monthly_K = tas_ssp3_slice['tas'].groupby("time.month").mean('time',keep_attrs=True)
tas_ssp5_mean_monthly_K = tas_ssp5_slice['tas'].groupby("time.month").mean('time',keep_attrs=True)

# Convert from Kelvin to Celsius
tas_hist_mean_monthly_C = tas_hist_mean_monthly_K-273.15
tas_ssp1_mean_monthly_C = tas_ssp1_mean_monthly_K-273.15
tas_ssp2_mean_monthly_C = tas_ssp2_mean_monthly_K-273.15
tas_ssp3_mean_monthly_C = tas_ssp3_mean_monthly_K-273.15
tas_ssp5_mean_monthly_C = tas_ssp5_mean_monthly_K-273.15


# Precipitation files
# Convert from precipitation flux (mm per second) to daily totals (mm per 24 hours)
# 60 seconds x 60 minutes x 24 hours = 86,400 seconds in one day
pre_hist_day = pre_hist_slice['pr'] * 86400.
pre_ssp1_day = pre_ssp1_slice['pr'] * 86400.
pre_ssp2_day = pre_ssp2_slice['pr'] * 86400.
pre_ssp3_day = pre_ssp3_slice['pr'] * 86400.
pre_ssp5_day = pre_ssp5_slice['pr'] * 86400.

# Assign the number of days in each month
# adapted from http://xarray.pydata.org/en/stable/examples/monthly-means.html
# Build a dictionary of calendars
dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}

# Define new function get_dpm to retrieve the length of each month
def get_dpm(time, calendar='standard'):
    #return an array of days per month corresponding to the months provided in `months`
    month_length = np.zeros(len(time), dtype=np.int)
    cal_days = dpm[calendar]
    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
    return month_length

# Use get_dpm to assign the number of days in each month to the sliced dataset
pre_hist_month_length = xr.DataArray(get_dpm(pre_hist_day.time.to_index(), calendar='noleap'), coords=[pre_hist_day.time], name='month_length')
pre_ssp1_month_length = xr.DataArray(get_dpm(pre_ssp1_day.time.to_index(), calendar='noleap'), coords=[pre_ssp1_day.time], name='month_length')
pre_ssp2_month_length = xr.DataArray(get_dpm(pre_ssp2_day.time.to_index(), calendar='noleap'), coords=[pre_ssp2_day.time], name='month_length')
pre_ssp3_month_length = xr.DataArray(get_dpm(pre_ssp3_day.time.to_index(), calendar='noleap'), coords=[pre_ssp3_day.time], name='month_length')
pre_ssp5_month_length = xr.DataArray(get_dpm(pre_ssp5_day.time.to_index(), calendar='noleap'), coords=[pre_ssp5_day.time], name='month_length')

# Multiply the number of days in each month by the mean daily precipitation values (mm per day)
pre_hist_mth= pre_hist_month_length * pre_hist_day
pre_ssp1_mth= pre_ssp1_month_length * pre_ssp1_day
pre_ssp2_mth= pre_ssp2_month_length * pre_ssp2_day
pre_ssp3_mth= pre_ssp3_month_length * pre_ssp3_day
pre_ssp5_mth= pre_ssp5_month_length * pre_ssp5_day

# Calculate mean precipitation values for each month
# GroupBy subdivides dataset into months before averaging. This code produces mean monthly precipitation totals.
pre_hist_mean_mth = pre_hist_mth.groupby("time.month").mean('time',keep_attrs=True)
pre_ssp1_mean_mth = pre_ssp1_mth.groupby("time.month").mean('time',keep_attrs=True)
pre_ssp2_mean_mth = pre_ssp2_mth.groupby("time.month").mean('time',keep_attrs=True)
pre_ssp3_mean_mth = pre_ssp3_mth.groupby("time.month").mean('time',keep_attrs=True)
pre_ssp5_mean_mth = pre_ssp5_mth.groupby("time.month").mean('time',keep_attrs=True)

print('Step 2: Calculate Monthly Averages Complete')

########################################################################################################
"""
STEP 3: MASK CMIP OCEANIC CELLS 
"""
########################################################################################################

"""
(3.1) Use xarray to assign the land cover percentage data for the CMIP model to a new object.
"""
print('(3.1) Checking CMIP land-sea mask...')

#Use xarray to open the mask dataset
mask_dset = xr.open_dataset(mask_file)
# Assign the land percentage variable ['sftlf'] to a new object
land_perc = mask_dset['sftlf']
# check that max land area is 100 % (if values range from 0 - 1 adjust code in section below)
print('Max land area (CMIP mask):', land_perc.data.max(), '%')
# check that min land area is 0 %
print('Min land area (CMIP mask):', land_perc.data.min(), '%') 

"""
(3.2) Mask out ocean in CMIP datasets (i.e. selecting only grid cells with > 50 % land)
"""
print('(3.2) Applying land-sea mask...')
#numpy includes a np.where function that allows us to simply use a logical command to mask out temperature data for oceanic grid cells (e.g. where land % is less than 50 %)
# if values range from 0 - 1 adjust conditional function
tas_hist_land_C = tas_hist_mean_monthly_C.where(land_perc.data > 50.) 
tas_ssp1_land_C = tas_ssp1_mean_monthly_C.where(land_perc.data > 50.) 
tas_ssp2_land_C = tas_ssp2_mean_monthly_C.where(land_perc.data > 50.)
tas_ssp3_land_C = tas_ssp3_mean_monthly_C.where(land_perc.data > 50.) 
tas_ssp5_land_C = tas_ssp5_mean_monthly_C.where(land_perc.data > 50.) 

# Mask out preciptiation data
pre_hist_land = pre_hist_mean_mth.where(land_perc.data > 50.) 
pre_ssp1_land = pre_ssp1_mean_mth.where(land_perc.data > 50.) 
pre_ssp2_land = pre_ssp2_mean_mth.where(land_perc.data > 50.) 
pre_ssp3_land = pre_ssp3_mean_mth.where(land_perc.data > 50.) 
pre_ssp5_land = pre_ssp5_mean_mth.where(land_perc.data > 50.) 

print('Step 3: Mask Out Ocean complete')

########################################################################################################
"""
STEP 4: EXTRAPOLATION 
Use Poisson Equation solver with overrelaxation to extrapolate terrestrial data over ocean.
"""
########################################################################################################
"""
(4.1) Convert xarray to iris cubes
"""
print ('(4.1) Converting data to iris cubes...')
# Temperature files
# Convert to iris from xarray:
tas_hist_land_Ciris = tas_hist_land_C.to_iris() 
tas_ssp1_land_Ciris = tas_ssp1_land_C.to_iris() 
tas_ssp2_land_Ciris = tas_ssp2_land_C.to_iris() 
tas_ssp3_land_Ciris = tas_ssp3_land_C.to_iris() 
tas_ssp5_land_Ciris = tas_ssp5_land_C.to_iris() 
# Add the cyclic attribute to the iris cube to ensure data covers full 360 degrees of longitude
tas_hist_land_Ciris.coord('longitude').circular = True
tas_ssp1_land_Ciris.coord('longitude').circular = True
tas_ssp2_land_Ciris.coord('longitude').circular = True
tas_ssp3_land_Ciris.coord('longitude').circular = True
tas_ssp5_land_Ciris.coord('longitude').circular = True

# Precipitation files
# Convert to iris from xarray:
pre_hist_land_iris = pre_hist_land.to_iris() 
pre_ssp1_land_iris = pre_ssp1_land.to_iris() 
pre_ssp2_land_iris = pre_ssp2_land.to_iris() 
pre_ssp3_land_iris = pre_ssp3_land.to_iris() 
pre_ssp5_land_iris = pre_ssp5_land.to_iris() 
# Add the cyclic attribute to the iris cube to ensure data covers full 360 degrees of longitude
pre_hist_land_iris.coord('longitude').circular = True
pre_ssp1_land_iris.coord('longitude').circular = True
pre_ssp2_land_iris.coord('longitude').circular = True
pre_ssp3_land_iris.coord('longitude').circular = True
pre_ssp5_land_iris.coord('longitude').circular = True

"""
(4.2) Conduct extrapolation procedure
"""
print ('(4.2) Extrapolating terrestrial climate over ocean...')
# Next step is to backfill (extrapolate) terrestrial climate over the ocean using gridfill.fill function
# Users can install gridfill manually from https://github.com/ajdawson/gridfill

## Definitions of terms used in Poisson gridfilling [and default values used in Morris et al. 2018]:
# eps = Tolerance for determining the solution complete. [1e-3]
# relax = Relaxation constant. Usually 0.45 <= *relax* <= 0.6. Defaults to 0.6. [0.6]
# itermax = Maximum number of iterations of the relaxation scheme. Defaults to 100 iterations. [2000] [here we specify a higher threshold e.g. 12000]
# initzonal = If *False* missing values will be initialized to zero, if *True* missing values will be initialized to the zonal mean. Defaultsto *False*. [True]
# cyclic = Set to *False* if the x-coordinate of the grid is not cyclic, set to *True* if it is cyclic. Defaults to *False*. [Not used]
# verbose = If *True* information about algorithm performance will be printed to stdout, if *False* nothing is printed. Defaults to *False*. [True]

# Temperature files
tas_hist_land_Ciris_backfilled = gridfill.fill_cube(tas_hist_land_Ciris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
tas_ssp1_land_Ciris_backfilled = gridfill.fill_cube(tas_ssp1_land_Ciris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
tas_ssp2_land_Ciris_backfilled = gridfill.fill_cube(tas_ssp2_land_Ciris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
tas_ssp3_land_Ciris_backfilled = gridfill.fill_cube(tas_ssp3_land_Ciris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
tas_ssp5_land_Ciris_backfilled = gridfill.fill_cube(tas_ssp5_land_Ciris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)

# Precipitation files
pre_hist_land_iris_backfilled = gridfill.fill_cube(pre_hist_land_iris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
pre_ssp1_land_iris_backfilled = gridfill.fill_cube(pre_ssp1_land_iris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
pre_ssp2_land_iris_backfilled = gridfill.fill_cube(pre_ssp2_land_iris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
pre_ssp3_land_iris_backfilled = gridfill.fill_cube(pre_ssp3_land_iris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)
pre_ssp5_land_iris_backfilled = gridfill.fill_cube(pre_ssp5_land_iris, 1e-3, 0.6, 12000, initzonal=True, verbose=True)

print('Step 4: Extrapolation complete')

########################################################################################################
"""
STEP 5: REGRIDDING TO HIGHER RESOLUTION
Regrid (interpolate) to the same resolution as the CRU grid (0.5 x 0.5 deg). 
"""
########################################################################################################
"""
(5.1) Convert observational dataset to iris and reshape
"""
print('(5.1) Converting observational data to iris cube...')

# Temporarily convert CRU data to iris. Select only the 'tmp' variable (not stn). Use of either the tmp or pre grid is equivalent. 
CRU_tmp_array = iris.load(CRU_tmp_file)[1] # only tmp not stn

# OPTIONAL] Select desired dates 
import datetime
date1 = datetime.datetime.strptime('19610116T0000Z','%Y%m%dT%H%MZ')
date2 = datetime.datetime.strptime('19901216T0000Z','%Y%m%dT%H%MZ')
date_range = iris.Constraint(time=lambda cell: date1 <= cell.point <= date2 )
CRU_tmp_array_slice = CRU_tmp_array.extract(date_range)

"""
(5.2) Perform bilinear interpolation
"""
print("(5) Performing linear interpolation with Iris...")
# Here, we prefer the use of bilinear interpolation to regrid the CMIP climate data onto the CRU grid.
# Temperature files
tas_hist_land_Ciris_backfilled_high = tas_hist_land_Ciris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
tas_ssp1_land_Ciris_backfilled_high = tas_ssp1_land_Ciris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
tas_ssp2_land_Ciris_backfilled_high = tas_ssp2_land_Ciris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
tas_ssp3_land_Ciris_backfilled_high = tas_ssp3_land_Ciris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
tas_ssp5_land_Ciris_backfilled_high = tas_ssp5_land_Ciris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())

# Precipitation files
pre_hist_land_iris_backfilled_high = pre_hist_land_iris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
pre_ssp1_land_iris_backfilled_high = pre_ssp1_land_iris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
pre_ssp2_land_iris_backfilled_high = pre_ssp2_land_iris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
pre_ssp3_land_iris_backfilled_high = pre_ssp3_land_iris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())
pre_ssp5_land_iris_backfilled_high = pre_ssp5_land_iris_backfilled.regrid(CRU_tmp_array_slice, iris.analysis.Linear())

"""
(5.3)  Convert back to xarray
"""
print('(5.3) Converting interpolated data back to xarray...')

## Convert back to xarray:
# Temperature
tas_hist_land_C_backfilled_high = xr.DataArray.from_iris(tas_hist_land_Ciris_backfilled_high)
tas_ssp1_land_C_backfilled_high = xr.DataArray.from_iris(tas_ssp1_land_Ciris_backfilled_high)
tas_ssp2_land_C_backfilled_high = xr.DataArray.from_iris(tas_ssp2_land_Ciris_backfilled_high)
tas_ssp3_land_C_backfilled_high = xr.DataArray.from_iris(tas_ssp3_land_Ciris_backfilled_high)
tas_ssp5_land_C_backfilled_high = xr.DataArray.from_iris(tas_ssp5_land_Ciris_backfilled_high)

# Precipitation
pre_hist_land_backfilled_high = xr.DataArray.from_iris(pre_hist_land_iris_backfilled_high)
pre_ssp1_land_backfilled_high = xr.DataArray.from_iris(pre_ssp1_land_iris_backfilled_high)
pre_ssp2_land_backfilled_high = xr.DataArray.from_iris(pre_ssp2_land_iris_backfilled_high)
pre_ssp3_land_backfilled_high = xr.DataArray.from_iris(pre_ssp3_land_iris_backfilled_high)
pre_ssp5_land_backfilled_high = xr.DataArray.from_iris(pre_ssp5_land_iris_backfilled_high)

print("Step 5: Interpolation complete")

########################################################################################################
"""
STEP 6: APPLY OBSERVATIONAL LAND-SEA MASK
"""
########################################################################################################
"""
(6.1) Create CRU land sea mask
"""
print("(6.1) Creating CRU land-sea mask...")

## Reload CRU data as an xarray, select climate variable, and reshape to match shape of CMIP array.
# Temperature files
CRU_tmp_slice_xr = CRU_tmp_dset.sel(time=slice("1961-01-16", "1990-12-16"))
CRU_tmp_xr = CRU_tmp_slice_xr['tmp'].groupby("time.month").mean('time',keep_attrs=True)

# Precipitation files
CRU_pre_slice_xr = CRU_pre_dset.sel(time=slice("1961-01-16", "1990-12-16"))
CRU_pre_xr = CRU_pre_slice_xr['pre'].groupby("time.month").mean('time',keep_attrs=True)

"""
(6.2) Apply CRU land sea mask to downscaled CMIP data
"""
print("(6) Applying Observational Mask")

# The fill value for missing values (i.e. oceanic cells) in the CRU data is -999. This line selects only those which are greater than that value (i.e. terrestrial cells).
# Temperature files
tas_hist_land_C_backfilled_high_masked = tas_hist_land_C_backfilled_high.where(CRU_tmp_xr.data >-998) 
tas_ssp1_land_C_backfilled_high_masked = tas_ssp1_land_C_backfilled_high.where(CRU_tmp_xr.data >-998) 
tas_ssp2_land_C_backfilled_high_masked = tas_ssp2_land_C_backfilled_high.where(CRU_tmp_xr.data >-998) 
tas_ssp3_land_C_backfilled_high_masked = tas_ssp3_land_C_backfilled_high.where(CRU_tmp_xr.data >-998) 
tas_ssp5_land_C_backfilled_high_masked = tas_ssp5_land_C_backfilled_high.where(CRU_tmp_xr.data >-998) 

# Precipitation files
pre_hist_land_backfilled_high_masked = pre_hist_land_backfilled_high.where(CRU_tmp_xr.data >-998) 
pre_ssp1_land_backfilled_high_masked = pre_ssp1_land_backfilled_high.where(CRU_tmp_xr.data >-998) 
pre_ssp2_land_backfilled_high_masked = pre_ssp2_land_backfilled_high.where(CRU_tmp_xr.data >-998) 
pre_ssp3_land_backfilled_high_masked = pre_ssp3_land_backfilled_high.where(CRU_tmp_xr.data >-998) 
pre_ssp5_land_backfilled_high_masked = pre_ssp5_land_backfilled_high.where(CRU_tmp_xr.data >-998) 

print("Step 6: Apply Observational Land-Sea Mask complete")

########################################################################################################
"""
STEP 7: BIAS CORRECTION
"""
########################################################################################################
"""
(7.1) Variable formatting
"""
print('(7.1) Preparing data for bias-correction...')

# Temperature
CRU_tmp = CRU_tmp_xr
CMIP_hist_tmp = tas_hist_land_C_backfilled_high_masked
SSP1_tmp = tas_ssp1_land_C_backfilled_high_masked
SSP2_tmp = tas_ssp2_land_C_backfilled_high_masked
SSP3_tmp = tas_ssp3_land_C_backfilled_high_masked
SSP5_tmp = tas_ssp5_land_C_backfilled_high_masked

# Precipitation
CRU_pre = CRU_pre_xr
CMIP_hist_pre = pre_hist_land_backfilled_high_masked
SSP1_pre = pre_ssp1_land_backfilled_high_masked
SSP2_pre = pre_ssp2_land_backfilled_high_masked
SSP3_pre = pre_ssp3_land_backfilled_high_masked
SSP5_pre = pre_ssp5_land_backfilled_high_masked

"""
(7.2) Temperature Bias correction

BCor_Temp = (CMIP_fut_tmp - CMIP_hist_tmp) + CRU_tmp
"""
print('(7.2) Performing temperature bias-correction...')

# Temperature correction calculation is followed by a function that renames each variable:
SSP1_BCor_tmp = (SSP1_tmp - CMIP_hist_tmp) + CRU_tmp
SSP1_BCor_tmp = SSP1_BCor_tmp.rename('mean monthly near-surface temperature (degrees Celsius)')
SSP2_BCor_tmp = (SSP2_tmp - CMIP_hist_tmp) + CRU_tmp
SSP2_BCor_tmp = SSP2_BCor_tmp.rename('mean monthly near-surface temperature (degrees Celsius)') 
SSP3_BCor_tmp = (SSP3_tmp - CMIP_hist_tmp) + CRU_tmp
SSP3_BCor_tmp = SSP3_BCor_tmp.rename('mean monthly near-surface temperature (degrees Celsius)') 
SSP5_BCor_tmp = (SSP5_tmp - CMIP_hist_tmp) + CRU_tmp
SSP5_BCor_tmp = SSP5_BCor_tmp.rename('mean monthly near-surface temperature (degrees Celsius)')

"""
(7.3) Precipitation Bias correction

BCor_Pre = (CRU_pre / CMIP_hist_pre) * CMIP_fut_pre
"""
print('(7.3) Performing precipitation bias-correction...')


# Precipitation correction calculations:
# First calculate alpha...
a = CRU_pre / CMIP_hist_pre
# Set limits for alpha (from Morris et al. 2018)...
a = xr.where(a < 0.25, 0.25, a)
a_ltd = xr.where(a > 4.0, 4.0, a)

# Apply the limited alpha coefficient to bias correct future precipitation
SSP1_BCor_pre = a_ltd * SSP1_pre
SSP1_BCor_pre = SSP1_BCor_pre.rename('mean monthly precipitation (mm)') 
SSP2_BCor_pre = a_ltd * SSP2_pre
SSP2_BCor_pre = SSP2_BCor_pre.rename('mean monthly precipitation (mm)')
SSP3_BCor_pre = a_ltd * SSP3_pre
SSP3_BCor_pre = SSP3_BCor_pre.rename('mean monthly precipitation (mm)') 
SSP5_BCor_pre = a_ltd * SSP5_pre
SSP5_BCor_pre = SSP5_BCor_pre.rename('mean monthly precipitation (mm)') 

"""
(7.4) Subset to the region of interest
"""
# (OPTIONAL) This function selects only grid cells that fall within the latitudinal bands specified
# [ANSWER EITHER: yes OR no]
answer = input('(7.4) [OPTIONAL] Crop output to study region?:')
if answer.lower().startswith("y"):
      # Temperature
      CMIP_hist_tmp = CMIP_hist_tmp.sel(lat=slice(44., 90.))
      SSP1_BCor_tmp = SSP1_BCor_tmp.sel(lat=slice(44., 90.))
      SSP2_BCor_tmp = SSP2_BCor_tmp.sel(lat=slice(44., 90.))
      SSP3_BCor_tmp = SSP3_BCor_tmp.sel(lat=slice(44., 90.))
      SSP5_BCor_tmp = SSP5_BCor_tmp.sel(lat=slice(44., 90.))

      # Preciptiation
      CMIP_hist_pre = CMIP_hist_pre.sel(lat=slice(44., 90.))
      SSP1_BCor_pre = SSP1_BCor_pre.sel(lat=slice(44., 90.))
      SSP2_BCor_pre = SSP2_BCor_pre.sel(lat=slice(44., 90.))
      SSP3_BCor_pre = SSP3_BCor_pre.sel(lat=slice(44., 90.))
      SSP5_BCor_pre = SSP5_BCor_pre.sel(lat=slice(44., 90.))

      tmp_DIR= tmp_DIR+'sliced_'
      pre_DIR= pre_DIR+'sliced_'
      
elif answer.lower().startswith("n"):
    
      tmp_DIR= tmp_DIR+'global_'
      pre_DIR= pre_DIR+'global_'

else:
        print("Enter either yes/no")

print("Step 7 complete")

########################################################################################################
"""
STEP 8: OUTPUT RESULTS 
"""
########################################################################################################
"""
(8.1) Exporting the results to netcdf format
"""

import sys
# Optional choice to export as netcdf or pass
answer = input('(8.1) [OPTIONAL] Export data files to netCDF?:')
if answer.lower().startswith("y"):
      print("(8.1) Data export to NetCDF...")
      # Prevent warnings from flashing up - turn off/on as desired
      # Turned off as no issue with 'true divide' (dividing by NaN).
      np.warnings.filterwarnings('ignore')
      # Temperature files
      CMIP_hist_tmp.to_netcdf(tmp_DIR+'cmip_hist.nc')
      print('tmp_CMIP_hist.nc complete')
      CRU_tmp.to_netcdf(tmp_DIR+'cru_1961_1990.nc')
      print('tmp_CRU_1961_1990.nc complete')
      SSP1_BCor_tmp.to_netcdf(tmp_DIR+'bcor_ssp1.nc')
      print('tmp_bcor_ssp1.nc complete')
      SSP2_BCor_tmp.to_netcdf(tmp_DIR+'bcor_ssp2.nc')
      print('tmp_bcor_ssp2.nc complete')
      SSP3_BCor_tmp.to_netcdf(tmp_DIR+'bcor_ssp3.nc')
      print('tmp_bcor_ssp3.nc complete')
      SSP5_BCor_tmp.to_netcdf(tmp_DIR+'bcor_ssp5.nc')
      print('tmp_bcor_ssp5.nc complete')
      # Precipitation files
      CMIP_hist_pre.to_netcdf(pre_DIR+'cmip_hist.nc')
      print('pre_CMIP_hist.nc complete')
      CRU_pre.to_netcdf(pre_DIR+'cru_1961_1990.nc')
      print('pre_CRU_1961_1990.nc complete')
      SSP1_BCor_pre.to_netcdf(pre_DIR+'bcor_ssp1.nc')
      print('pre_bcor_ssp1.nc complete')
      SSP2_BCor_pre.to_netcdf(pre_DIR+'bcor_ssp2.nc')
      print('pre_bcor_ssp2.nc complete')
      SSP3_BCor_pre.to_netcdf(pre_DIR+'bcor_ssp3.nc')
      print('pre_bcor_ssp3.nc complete')
      SSP5_BCor_pre.to_netcdf(pre_DIR+'bcor_ssp5.nc')
      print('pre_bcor_ssp5.nc complete')
      # Turn warnings back on
      np.warnings.filterwarnings('default')
elif answer.lower().startswith("n"):
    pass
else:
        print("Enter either yes/no")

"""
(8.2) Export the results as .csv
"""
# Optional choice to export as netcdf or pass
answer = input('(8.2) [OPTIONAL] Export data files to .csv?:')
if answer.lower().startswith("y"):
      print("(8.2) Data export to .csv...")
# Prevent warnings from flashing up - turn off/on as desired
      np.warnings.filterwarnings('ignore')
# Temperature files
# Historical
      CMIP_hist_tmp = CMIP_hist_tmp.rename('mean monthly near-surface temperature (degrees Celsius)') # ensure dataset is named
      hist_tmp_jan = CMIP_hist_tmp.sel(month=1) # select tas data from the first month
      hist_tmp_jan_df= hist_tmp_jan.to_dataframe() # turn this data into a pandas dataframe
      hist_tmp_jan_df = hist_tmp_jan_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jan_MMT"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      hist_tmp_feb = CMIP_hist_tmp.sel(month=2)
      hist_tmp_feb_df= hist_tmp_feb.to_dataframe()
      hist_tmp_feb_df = hist_tmp_feb_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Feb_MMT"})
      print('##')      
      hist_tmp_mar = CMIP_hist_tmp.sel(month=3)
      hist_tmp_mar_df= hist_tmp_mar.to_dataframe()
      hist_tmp_mar_df = hist_tmp_mar_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Mar_MMT"})
      print('###')      
      hist_tmp_apr = CMIP_hist_tmp.sel(month=4)
      hist_tmp_apr_df= hist_tmp_apr.to_dataframe()
      hist_tmp_apr_df = hist_tmp_apr_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Apr_MMT"})
      print('####')      
      hist_tmp_may = CMIP_hist_tmp.sel(month=5)
      hist_tmp_may_df= hist_tmp_may.to_dataframe()
      hist_tmp_may_df = hist_tmp_may_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "May_MMT"})
      print('#####')      
      hist_tmp_jun = CMIP_hist_tmp.sel(month=6)
      hist_tmp_jun_df= hist_tmp_jun.to_dataframe()
      hist_tmp_jun_df = hist_tmp_jun_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jun_MMT"})
      print('######')      
      hist_tmp_jul = CMIP_hist_tmp.sel(month=7)
      hist_tmp_jul_df= hist_tmp_jul.to_dataframe()
      hist_tmp_jul_df = hist_tmp_jul_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jul_MMT"})
      print('#######')      
      hist_tmp_aug = CMIP_hist_tmp.sel(month=8)
      hist_tmp_aug_df= hist_tmp_aug.to_dataframe()
      hist_tmp_aug_df = hist_tmp_aug_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Aug_MMT"})
      print('########')      
      hist_tmp_sep = CMIP_hist_tmp.sel(month=9)
      hist_tmp_sep_df= hist_tmp_sep.to_dataframe()
      hist_tmp_sep_df = hist_tmp_sep_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Sep_MMT"})
      print('#########')      
      hist_tmp_oct = CMIP_hist_tmp.sel(month=10)
      hist_tmp_oct_df= hist_tmp_oct.to_dataframe()
      hist_tmp_oct_df = hist_tmp_oct_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Oct_MMT"})
      print('##########')      
      hist_tmp_nov = CMIP_hist_tmp.sel(month=11)
      hist_tmp_nov_df= hist_tmp_nov.to_dataframe()
      hist_tmp_nov_df = hist_tmp_nov_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Nov_MMT"})
      print('###########')      
      hist_tmp_dec = CMIP_hist_tmp.sel(month=12)
      hist_tmp_dec_df= hist_tmp_dec.to_dataframe()
      hist_tmp_dec_df = hist_tmp_dec_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Dec_MMT"})
      print('############')      
      hist_tmp_df = pd.concat([hist_tmp_jan_df, hist_tmp_feb_df, hist_tmp_mar_df, hist_tmp_apr_df, hist_tmp_may_df, hist_tmp_jun_df, hist_tmp_jul_df, hist_tmp_aug_df, hist_tmp_sep_df, hist_tmp_oct_df, hist_tmp_nov_df, hist_tmp_dec_df], axis=1) # add each variable as a column
      hist_tmp_df = hist_tmp_df.reset_index() # add id column
      hist_tmp_df.index = hist_tmp_df.index + 1 # start id index at 1, not 0
      hist_tmp_df.to_csv(tmp_DIR+'CMIP_hist.csv')
      print('CMIP_hist_tmp.csv complete')


# SSP1 tmp
      SSP1_tmp_jan = SSP1_BCor_tmp.sel(month=1) # select tas data from the first month
      SSP1_tmp_jan_df= SSP1_tmp_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP1_tmp_jan_df = SSP1_tmp_jan_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jan_MMT"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP1_tmp_feb = SSP1_BCor_tmp.sel(month=2)
      SSP1_tmp_feb_df= SSP1_tmp_feb.to_dataframe()
      SSP1_tmp_feb_df = SSP1_tmp_feb_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Feb_MMT"})
      print('##')
      SSP1_tmp_mar = SSP1_BCor_tmp.sel(month=3)
      SSP1_tmp_mar_df= SSP1_tmp_mar.to_dataframe()
      SSP1_tmp_mar_df = SSP1_tmp_mar_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Mar_MMT"})
      print('###')
      SSP1_tmp_apr = SSP1_BCor_tmp.sel(month=4)
      SSP1_tmp_apr_df= SSP1_tmp_apr.to_dataframe()
      SSP1_tmp_apr_df = SSP1_tmp_apr_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Apr_MMT"})
      print('####')
      SSP1_tmp_may = SSP1_BCor_tmp.sel(month=5)
      SSP1_tmp_may_df= SSP1_tmp_may.to_dataframe()
      SSP1_tmp_may_df = SSP1_tmp_may_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "May_MMT"})
      print('#####')
      SSP1_tmp_jun = SSP1_BCor_tmp.sel(month=6)
      SSP1_tmp_jun_df= SSP1_tmp_jun.to_dataframe()
      SSP1_tmp_jun_df = SSP1_tmp_jun_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jun_MMT"})
      print('######')
      SSP1_tmp_jul = SSP1_BCor_tmp.sel(month=7)
      SSP1_tmp_jul_df= SSP1_tmp_jul.to_dataframe()
      SSP1_tmp_jul_df = SSP1_tmp_jul_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jul_MMT"})
      print('#######')
      SSP1_tmp_aug = SSP1_BCor_tmp.sel(month=8)
      SSP1_tmp_aug_df= SSP1_tmp_aug.to_dataframe()
      SSP1_tmp_aug_df = SSP1_tmp_aug_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Aug_MMT"})
      print('########')
      SSP1_tmp_sep = SSP1_BCor_tmp.sel(month=9)
      SSP1_tmp_sep_df= SSP1_tmp_sep.to_dataframe()
      SSP1_tmp_sep_df = SSP1_tmp_sep_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Sep_MMT"})
      print('#########')
      SSP1_tmp_oct = SSP1_BCor_tmp.sel(month=10)
      SSP1_tmp_oct_df= SSP1_tmp_oct.to_dataframe()
      SSP1_tmp_oct_df = SSP1_tmp_oct_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Oct_MMT"})
      print('##########')
      SSP1_tmp_nov = SSP1_BCor_tmp.sel(month=11)
      SSP1_tmp_nov_df= SSP1_tmp_nov.to_dataframe()
      SSP1_tmp_nov_df = SSP1_tmp_nov_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Nov_MMT"})
      print('###########')
      SSP1_tmp_dec = SSP1_BCor_tmp.sel(month=12)
      SSP1_tmp_dec_df= SSP1_tmp_dec.to_dataframe()
      SSP1_tmp_dec_df = SSP1_tmp_dec_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Dec_MMT"})
      print('############')
      SSP1_tmp_df = pd.concat([SSP1_tmp_jan_df, SSP1_tmp_feb_df, SSP1_tmp_mar_df, SSP1_tmp_apr_df, SSP1_tmp_may_df, SSP1_tmp_jun_df, SSP1_tmp_jul_df, SSP1_tmp_aug_df, SSP1_tmp_sep_df, SSP1_tmp_oct_df, SSP1_tmp_nov_df, SSP1_tmp_dec_df], axis=1) # add each variable as a column
      SSP1_tmp_df = SSP1_tmp_df.reset_index() # add id column
      SSP1_tmp_df.index = SSP1_tmp_df.index + 1 # start id index at 1, not 0
      SSP1_tmp_df.to_csv(tmp_DIR+'bcor_ssp1.csv')
      print('SSP1_tmp.csv complete')


# SSP2 tmp
      SSP2_tmp_jan = SSP2_BCor_tmp.sel(month=1) # select tas data from the first month
      SSP2_tmp_jan_df= SSP2_tmp_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP2_tmp_jan_df = SSP2_tmp_jan_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jan_MMT"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP2_tmp_feb = SSP2_BCor_tmp.sel(month=2)
      SSP2_tmp_feb_df= SSP2_tmp_feb.to_dataframe()
      SSP2_tmp_feb_df = SSP2_tmp_feb_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Feb_MMT"})
      print('##')
      SSP2_tmp_mar = SSP2_BCor_tmp.sel(month=3)
      SSP2_tmp_mar_df= SSP2_tmp_mar.to_dataframe()
      SSP2_tmp_mar_df = SSP2_tmp_mar_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Mar_MMT"})
      print('###')
      SSP2_tmp_apr = SSP2_BCor_tmp.sel(month=4)
      SSP2_tmp_apr_df= SSP2_tmp_apr.to_dataframe()
      SSP2_tmp_apr_df = SSP2_tmp_apr_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Apr_MMT"})
      print('####')
      SSP2_tmp_may = SSP2_BCor_tmp.sel(month=5)
      SSP2_tmp_may_df= SSP2_tmp_may.to_dataframe()
      SSP2_tmp_may_df = SSP2_tmp_may_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "May_MMT"})
      print('#####')
      SSP2_tmp_jun = SSP2_BCor_tmp.sel(month=6)
      SSP2_tmp_jun_df= SSP2_tmp_jun.to_dataframe()
      SSP2_tmp_jun_df = SSP2_tmp_jun_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jun_MMT"})
      print('######')
      SSP2_tmp_jul = SSP2_BCor_tmp.sel(month=7)
      SSP2_tmp_jul_df= SSP2_tmp_jul.to_dataframe()
      SSP2_tmp_jul_df = SSP2_tmp_jul_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jul_MMT"})
      print('#######')
      SSP2_tmp_aug = SSP2_BCor_tmp.sel(month=8)
      SSP2_tmp_aug_df= SSP2_tmp_aug.to_dataframe()
      SSP2_tmp_aug_df = SSP2_tmp_aug_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Aug_MMT"})
      print('########')
      SSP2_tmp_sep = SSP2_BCor_tmp.sel(month=9)
      SSP2_tmp_sep_df= SSP2_tmp_sep.to_dataframe()
      SSP2_tmp_sep_df = SSP2_tmp_sep_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Sep_MMT"})
      print('#########')
      SSP2_tmp_oct = SSP2_BCor_tmp.sel(month=10)
      SSP2_tmp_oct_df= SSP2_tmp_oct.to_dataframe()
      SSP2_tmp_oct_df = SSP2_tmp_oct_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Oct_MMT"})
      print('##########')
      SSP2_tmp_nov = SSP2_BCor_tmp.sel(month=11)
      SSP2_tmp_nov_df= SSP2_tmp_nov.to_dataframe()
      SSP2_tmp_nov_df = SSP2_tmp_nov_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Nov_MMT"})
      print('###########')
      SSP2_tmp_dec = SSP2_BCor_tmp.sel(month=12)
      SSP2_tmp_dec_df= SSP2_tmp_dec.to_dataframe()
      SSP2_tmp_dec_df = SSP2_tmp_dec_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Dec_MMT"})
      print('############')
      SSP2_tmp_df = pd.concat([SSP2_tmp_jan_df, SSP2_tmp_feb_df, SSP2_tmp_mar_df, SSP2_tmp_apr_df, SSP2_tmp_may_df, SSP2_tmp_jun_df, SSP2_tmp_jul_df, SSP2_tmp_aug_df, SSP2_tmp_sep_df, SSP2_tmp_oct_df, SSP2_tmp_nov_df, SSP2_tmp_dec_df], axis=1) # add each variable as a column
      SSP2_tmp_df = SSP2_tmp_df.reset_index() # add id column
      SSP2_tmp_df.index = SSP2_tmp_df.index + 1 # start id index at 1, not 0
      SSP2_tmp_df.to_csv(tmp_DIR+'bcor_ssp2.csv')
      print('SSP2_tmp.csv complete')


# SSP3 tmp
      SSP3_tmp_jan = SSP3_BCor_tmp.sel(month=1) # select tas data from the first month
      SSP3_tmp_jan_df= SSP3_tmp_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP3_tmp_jan_df = SSP3_tmp_jan_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jan_MMT"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP3_tmp_feb = SSP3_BCor_tmp.sel(month=2)
      SSP3_tmp_feb_df= SSP3_tmp_feb.to_dataframe()
      SSP3_tmp_feb_df = SSP3_tmp_feb_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Feb_MMT"})
      print('##')
      SSP3_tmp_mar = SSP3_BCor_tmp.sel(month=3)
      SSP3_tmp_mar_df= SSP3_tmp_mar.to_dataframe()
      SSP3_tmp_mar_df = SSP3_tmp_mar_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Mar_MMT"})
      print('###')
      SSP3_tmp_apr = SSP3_BCor_tmp.sel(month=4)
      SSP3_tmp_apr_df= SSP3_tmp_apr.to_dataframe()
      SSP3_tmp_apr_df = SSP3_tmp_apr_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Apr_MMT"})
      print('####')
      SSP3_tmp_may = SSP3_BCor_tmp.sel(month=5)
      SSP3_tmp_may_df= SSP3_tmp_may.to_dataframe()
      SSP3_tmp_may_df = SSP3_tmp_may_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "May_MMT"})
      print('#####')
      SSP3_tmp_jun = SSP3_BCor_tmp.sel(month=6)
      SSP3_tmp_jun_df= SSP3_tmp_jun.to_dataframe()
      SSP3_tmp_jun_df = SSP3_tmp_jun_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jun_MMT"})
      print('######')
      SSP3_tmp_jul = SSP3_BCor_tmp.sel(month=7)
      SSP3_tmp_jul_df= SSP3_tmp_jul.to_dataframe()
      SSP3_tmp_jul_df = SSP3_tmp_jul_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jul_MMT"})
      print('#######')
      SSP3_tmp_aug = SSP3_BCor_tmp.sel(month=8)
      SSP3_tmp_aug_df= SSP3_tmp_aug.to_dataframe()
      SSP3_tmp_aug_df = SSP3_tmp_aug_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Aug_MMT"})
      print('########')
      SSP3_tmp_sep = SSP3_BCor_tmp.sel(month=9)
      SSP3_tmp_sep_df= SSP3_tmp_sep.to_dataframe()
      SSP3_tmp_sep_df = SSP3_tmp_sep_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Sep_MMT"})
      print('#########')
      SSP3_tmp_oct = SSP3_BCor_tmp.sel(month=10)
      SSP3_tmp_oct_df= SSP3_tmp_oct.to_dataframe()
      SSP3_tmp_oct_df = SSP3_tmp_oct_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Oct_MMT"})
      print('##########')
      SSP3_tmp_nov = SSP3_BCor_tmp.sel(month=11)
      SSP3_tmp_nov_df= SSP3_tmp_nov.to_dataframe()
      SSP3_tmp_nov_df = SSP3_tmp_nov_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Nov_MMT"})
      print('###########')
      SSP3_tmp_dec = SSP3_BCor_tmp.sel(month=12)
      SSP3_tmp_dec_df= SSP3_tmp_dec.to_dataframe()
      SSP3_tmp_dec_df = SSP3_tmp_dec_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Dec_MMT"})
      print('############')
      SSP3_tmp_df = pd.concat([SSP3_tmp_jan_df, SSP3_tmp_feb_df, SSP3_tmp_mar_df, SSP3_tmp_apr_df, SSP3_tmp_may_df, SSP3_tmp_jun_df, SSP3_tmp_jul_df, SSP3_tmp_aug_df, SSP3_tmp_sep_df, SSP3_tmp_oct_df, SSP3_tmp_nov_df, SSP3_tmp_dec_df], axis=1) # add each variable as a column
      SSP3_tmp_df = SSP3_tmp_df.reset_index() # add id column
      SSP3_tmp_df.index = SSP3_tmp_df.index + 1 # start id index at 1, not 0
      SSP3_tmp_df.to_csv(tmp_DIR+'bcor_ssp3.csv')
      print('SSP3_tmp.csv complete')
      

      # SSP5 tmp
      SSP5_tmp_jan = SSP5_BCor_tmp.sel(month=1) # select tas data from the first month
      SSP5_tmp_jan_df= SSP5_tmp_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP5_tmp_jan_df = SSP5_tmp_jan_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jan_MMT"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP5_tmp_feb = SSP5_BCor_tmp.sel(month=2)
      SSP5_tmp_feb_df= SSP5_tmp_feb.to_dataframe()
      SSP5_tmp_feb_df = SSP5_tmp_feb_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Feb_MMT"})
      print('##')
      SSP5_tmp_mar = SSP5_BCor_tmp.sel(month=3)
      SSP5_tmp_mar_df= SSP5_tmp_mar.to_dataframe()
      SSP5_tmp_mar_df = SSP5_tmp_mar_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Mar_MMT"})
      print('###')
      SSP5_tmp_apr = SSP5_BCor_tmp.sel(month=4)
      SSP5_tmp_apr_df= SSP5_tmp_apr.to_dataframe()
      SSP5_tmp_apr_df = SSP5_tmp_apr_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Apr_MMT"})
      print('####')
      SSP5_tmp_may = SSP5_BCor_tmp.sel(month=5)
      SSP5_tmp_may_df= SSP5_tmp_may.to_dataframe()
      SSP5_tmp_may_df = SSP5_tmp_may_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "May_MMT"})
      print('#####')
      SSP5_tmp_jun = SSP5_BCor_tmp.sel(month=6)
      SSP5_tmp_jun_df= SSP5_tmp_jun.to_dataframe()
      SSP5_tmp_jun_df = SSP5_tmp_jun_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jun_MMT"})
      print('######')
      SSP5_tmp_jul = SSP5_BCor_tmp.sel(month=7)
      SSP5_tmp_jul_df= SSP5_tmp_jul.to_dataframe()
      SSP5_tmp_jul_df = SSP5_tmp_jul_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Jul_MMT"})
      print('#######')
      SSP5_tmp_aug = SSP5_BCor_tmp.sel(month=8)
      SSP5_tmp_aug_df= SSP5_tmp_aug.to_dataframe()
      SSP5_tmp_aug_df = SSP5_tmp_aug_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Aug_MMT"})
      print('########')
      SSP5_tmp_sep = SSP5_BCor_tmp.sel(month=9)
      SSP5_tmp_sep_df= SSP5_tmp_sep.to_dataframe()
      SSP5_tmp_sep_df = SSP5_tmp_sep_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Sep_MMT"})
      print('#########')
      SSP5_tmp_oct = SSP5_BCor_tmp.sel(month=10)
      SSP5_tmp_oct_df= SSP5_tmp_oct.to_dataframe()
      SSP5_tmp_oct_df = SSP5_tmp_oct_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Oct_MMT"})
      print('##########')
      SSP5_tmp_nov = SSP5_BCor_tmp.sel(month=11)
      SSP5_tmp_nov_df= SSP5_tmp_nov.to_dataframe()
      SSP5_tmp_nov_df = SSP5_tmp_nov_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Nov_MMT"})
      print('###########')
      SSP5_tmp_dec = SSP5_BCor_tmp.sel(month=12)
      SSP5_tmp_dec_df= SSP5_tmp_dec.to_dataframe()
      SSP5_tmp_dec_df = SSP5_tmp_dec_df.drop(["month", "height"], axis=1).rename(columns={"mean monthly near-surface temperature (degrees Celsius)": "Dec_MMT"})
      print('############')
      SSP5_tmp_df = pd.concat([SSP5_tmp_jan_df, SSP5_tmp_feb_df, SSP5_tmp_mar_df, SSP5_tmp_apr_df, SSP5_tmp_may_df, SSP5_tmp_jun_df, SSP5_tmp_jul_df, SSP5_tmp_aug_df, SSP5_tmp_sep_df, SSP5_tmp_oct_df, SSP5_tmp_nov_df, SSP5_tmp_dec_df], axis=1) # add each variable as a column
      SSP5_tmp_df = SSP5_tmp_df.reset_index() # add id column
      SSP5_tmp_df.index = SSP5_tmp_df.index + 1 # start id index at 1, not 0
      SSP5_tmp_df.to_csv(tmp_DIR+'bcor_ssp5.csv')
      print('SSP5_tmp.csv complete')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
# Precipitation files
# Historical 
      CMIP_hist_pre = CMIP_hist_pre.rename('mean monthly precipitation (mm)') # ensure dataset is named
      hist_pre_jan = CMIP_hist_pre.sel(month=1) # select pre data from the first month
      hist_pre_jan_df= hist_pre_jan.to_dataframe() # turn this data into a pandas dataframe
      hist_pre_jan_df = hist_pre_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jan_pre"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      hist_pre_feb = CMIP_hist_pre.sel(month=2)
      hist_pre_feb_df= hist_pre_feb.to_dataframe()
      hist_pre_feb_df = hist_pre_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Feb_pre"})
      print('##')
      hist_pre_mar = CMIP_hist_pre.sel(month=3)
      hist_pre_mar_df= hist_pre_mar.to_dataframe()
      hist_pre_mar_df = hist_pre_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Mar_pre"})
      print('###')
      hist_pre_apr = CMIP_hist_pre.sel(month=4)
      hist_pre_apr_df= hist_pre_apr.to_dataframe()
      hist_pre_apr_df = hist_pre_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Apr_pre"})
      print('####')
      hist_pre_may = CMIP_hist_pre.sel(month=5)
      hist_pre_may_df= hist_pre_may.to_dataframe()
      hist_pre_may_df = hist_pre_may_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "May_pre"})
      print('#####')
      hist_pre_jun = CMIP_hist_pre.sel(month=6)
      hist_pre_jun_df= hist_pre_jun.to_dataframe()
      hist_pre_jun_df = hist_pre_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jun_pre"})
      print('######')
      hist_pre_jul = CMIP_hist_pre.sel(month=7)
      hist_pre_jul_df= hist_pre_jul.to_dataframe()
      hist_pre_jul_df = hist_pre_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jul_pre"})
      print('#######')
      hist_pre_aug = CMIP_hist_pre.sel(month=8)
      hist_pre_aug_df= hist_pre_aug.to_dataframe()
      hist_pre_aug_df = hist_pre_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Aug_pre"})
      print('########')
      hist_pre_sep = CMIP_hist_pre.sel(month=9)
      hist_pre_sep_df= hist_pre_sep.to_dataframe()
      hist_pre_sep_df = hist_pre_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Sep_pre"})
      print('#########')
      hist_pre_oct = CMIP_hist_pre.sel(month=10)
      hist_pre_oct_df= hist_pre_oct.to_dataframe()
      hist_pre_oct_df = hist_pre_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Oct_pre"})
      print('##########')
      hist_pre_nov = CMIP_hist_pre.sel(month=11)
      hist_pre_nov_df= hist_pre_nov.to_dataframe()
      hist_pre_nov_df = hist_pre_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Nov_pre"})
      print('###########')
      hist_pre_dec = CMIP_hist_pre.sel(month=12)
      hist_pre_dec_df= hist_pre_dec.to_dataframe()
      hist_pre_dec_df = hist_pre_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Dec_pre"})
      print('############')
      hist_pre_df = pd.concat([hist_pre_jan_df, hist_pre_feb_df, hist_pre_mar_df, hist_pre_apr_df, hist_pre_may_df, hist_pre_jun_df, hist_pre_jul_df, hist_pre_aug_df, hist_pre_sep_df, hist_pre_oct_df, hist_pre_nov_df, hist_pre_dec_df], axis=1) # add each variable as a column
      hist_pre_df = hist_pre_df.reset_index() # add id column
      hist_pre_df.index = hist_pre_df.index + 1 # start id index at 1, not 0
      hist_pre_df.to_csv(pre_DIR+'CMIP_hist.csv')
      print('CMIP_hist_pre.csv complete')
      

# SSP1 pre
      SSP1_pre_jan = SSP1_BCor_pre.sel(month=1) # select pre data from the first month
      SSP1_pre_jan_df= SSP1_pre_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP1_pre_jan_df = SSP1_pre_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jan_pre"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP1_pre_feb = SSP1_BCor_pre.sel(month=2)
      SSP1_pre_feb_df= SSP1_pre_feb.to_dataframe()
      SSP1_pre_feb_df = SSP1_pre_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Feb_pre"})
      print('##')
      SSP1_pre_mar = SSP1_BCor_pre.sel(month=3)
      SSP1_pre_mar_df= SSP1_pre_mar.to_dataframe()
      SSP1_pre_mar_df = SSP1_pre_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Mar_pre"})
      print('###')
      SSP1_pre_apr = SSP1_BCor_pre.sel(month=4)
      SSP1_pre_apr_df= SSP1_pre_apr.to_dataframe()
      SSP1_pre_apr_df = SSP1_pre_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Apr_pre"})
      print('####')
      SSP1_pre_may = SSP1_BCor_pre.sel(month=5)
      SSP1_pre_may_df= SSP1_pre_may.to_dataframe()
      SSP1_pre_may_df = SSP1_pre_may_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "May_pre"})
      print('#####')
      SSP1_pre_jun = SSP1_BCor_pre.sel(month=6)
      SSP1_pre_jun_df= SSP1_pre_jun.to_dataframe()
      SSP1_pre_jun_df = SSP1_pre_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jun_pre"})
      print('######')
      SSP1_pre_jul = SSP1_BCor_pre.sel(month=7)
      SSP1_pre_jul_df= SSP1_pre_jul.to_dataframe()
      SSP1_pre_jul_df = SSP1_pre_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jul_pre"})
      print('#######')
      SSP1_pre_aug = SSP1_BCor_pre.sel(month=8)
      SSP1_pre_aug_df= SSP1_pre_aug.to_dataframe()
      SSP1_pre_aug_df = SSP1_pre_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Aug_pre"})
      print('########')
      SSP1_pre_sep = SSP1_BCor_pre.sel(month=9)
      SSP1_pre_sep_df= SSP1_pre_sep.to_dataframe()
      SSP1_pre_sep_df = SSP1_pre_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Sep_pre"})
      print('#########')
      SSP1_pre_oct = SSP1_BCor_pre.sel(month=10)
      SSP1_pre_oct_df= SSP1_pre_oct.to_dataframe()
      SSP1_pre_oct_df = SSP1_pre_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Oct_pre"})
      print('##########')
      SSP1_pre_nov = SSP1_BCor_pre.sel(month=11)
      SSP1_pre_nov_df= SSP1_pre_nov.to_dataframe()
      SSP1_pre_nov_df = SSP1_pre_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Nov_pre"})
      print('###########')
      SSP1_pre_dec = SSP1_BCor_pre.sel(month=12)
      SSP1_pre_dec_df= SSP1_pre_dec.to_dataframe()
      SSP1_pre_dec_df = SSP1_pre_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Dec_pre"})
      print('############')
      SSP1_pre_df = pd.concat([SSP1_pre_jan_df, SSP1_pre_feb_df, SSP1_pre_mar_df, SSP1_pre_apr_df, SSP1_pre_may_df, SSP1_pre_jun_df, SSP1_pre_jul_df, SSP1_pre_aug_df, SSP1_pre_sep_df, SSP1_pre_oct_df, SSP1_pre_nov_df, SSP1_pre_dec_df], axis=1) # add each variable as a column
      SSP1_pre_df = SSP1_pre_df.reset_index() # add id column
      SSP1_pre_df.index = SSP1_pre_df.index + 1 # start id index at 1, not 0
      SSP1_pre_df.to_csv(pre_DIR+'bcor_SSP1.csv')
      print('SSP1_pre.csv complete')


# SSP2 pre
      SSP2_pre_jan = SSP2_BCor_pre.sel(month=1) # select pre data from the first month
      SSP2_pre_jan_df= SSP2_pre_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP2_pre_jan_df = SSP2_pre_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jan_pre"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP2_pre_feb = SSP2_BCor_pre.sel(month=2)
      SSP2_pre_feb_df= SSP2_pre_feb.to_dataframe()
      SSP2_pre_feb_df = SSP2_pre_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Feb_pre"})
      print('##')
      SSP2_pre_mar = SSP2_BCor_pre.sel(month=3)
      SSP2_pre_mar_df= SSP2_pre_mar.to_dataframe()
      SSP2_pre_mar_df = SSP2_pre_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Mar_pre"})
      print('###')
      SSP2_pre_apr = SSP2_BCor_pre.sel(month=4)
      SSP2_pre_apr_df= SSP2_pre_apr.to_dataframe()
      SSP2_pre_apr_df = SSP2_pre_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Apr_pre"})
      print('####')
      SSP2_pre_may = SSP2_BCor_pre.sel(month=5)
      SSP2_pre_may_df= SSP2_pre_may.to_dataframe()
      SSP2_pre_may_df = SSP2_pre_may_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "May_pre"})
      print('#####')
      SSP2_pre_jun = SSP2_BCor_pre.sel(month=6)
      SSP2_pre_jun_df= SSP2_pre_jun.to_dataframe()
      SSP2_pre_jun_df = SSP2_pre_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jun_pre"})
      print('######')
      SSP2_pre_jul = SSP2_BCor_pre.sel(month=7)
      SSP2_pre_jul_df= SSP2_pre_jul.to_dataframe()
      SSP2_pre_jul_df = SSP2_pre_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jul_pre"})
      print('#######')
      SSP2_pre_aug = SSP2_BCor_pre.sel(month=8)
      SSP2_pre_aug_df= SSP2_pre_aug.to_dataframe()
      SSP2_pre_aug_df = SSP2_pre_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Aug_pre"})
      print('########')
      SSP2_pre_sep = SSP2_BCor_pre.sel(month=9)
      SSP2_pre_sep_df= SSP2_pre_sep.to_dataframe()
      SSP2_pre_sep_df = SSP2_pre_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Sep_pre"})
      print('#########')
      SSP2_pre_oct = SSP2_BCor_pre.sel(month=10)
      SSP2_pre_oct_df= SSP2_pre_oct.to_dataframe()
      SSP2_pre_oct_df = SSP2_pre_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Oct_pre"})
      print('##########')
      SSP2_pre_nov = SSP2_BCor_pre.sel(month=11)
      SSP2_pre_nov_df= SSP2_pre_nov.to_dataframe()
      SSP2_pre_nov_df = SSP2_pre_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Nov_pre"})
      print('###########')
      SSP2_pre_dec = SSP2_BCor_pre.sel(month=12)
      SSP2_pre_dec_df= SSP2_pre_dec.to_dataframe()
      SSP2_pre_dec_df = SSP2_pre_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Dec_pre"})
      print('############')
      SSP2_pre_df = pd.concat([SSP2_pre_jan_df, SSP2_pre_feb_df, SSP2_pre_mar_df, SSP2_pre_apr_df, SSP2_pre_may_df, SSP2_pre_jun_df, SSP2_pre_jul_df, SSP2_pre_aug_df, SSP2_pre_sep_df, SSP2_pre_oct_df, SSP2_pre_nov_df, SSP2_pre_dec_df], axis=1) # add each variable as a column
      SSP2_pre_df = SSP2_pre_df.reset_index() # add id column
      SSP2_pre_df.index = SSP2_pre_df.index + 1 # start id index at 1, not 0
      SSP2_pre_df.to_csv(pre_DIR+'bcor_SSP2.csv')
      print('SSP2_pre.csv complete')


# SSP3 pre
      SSP3_pre_jan = SSP3_BCor_pre.sel(month=1) # select pre data from the first month
      SSP3_pre_jan_df= SSP3_pre_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP3_pre_jan_df = SSP3_pre_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jan_pre"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP3_pre_feb = SSP3_BCor_pre.sel(month=2)
      SSP3_pre_feb_df= SSP3_pre_feb.to_dataframe()
      SSP3_pre_feb_df = SSP3_pre_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Feb_pre"})
      print('##')
      SSP3_pre_mar = SSP3_BCor_pre.sel(month=3)
      SSP3_pre_mar_df= SSP3_pre_mar.to_dataframe()
      SSP3_pre_mar_df = SSP3_pre_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Mar_pre"})
      print('###')
      SSP3_pre_apr = SSP3_BCor_pre.sel(month=4)
      SSP3_pre_apr_df= SSP3_pre_apr.to_dataframe()
      SSP3_pre_apr_df = SSP3_pre_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Apr_pre"})
      print('####')
      SSP3_pre_may = SSP3_BCor_pre.sel(month=5)
      SSP3_pre_may_df= SSP3_pre_may.to_dataframe()
      SSP3_pre_may_df = SSP3_pre_may_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "May_pre"})
      print('#####')
      SSP3_pre_jun = SSP3_BCor_pre.sel(month=6)
      SSP3_pre_jun_df= SSP3_pre_jun.to_dataframe()
      SSP3_pre_jun_df = SSP3_pre_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jun_pre"})
      print('######')
      SSP3_pre_jul = SSP3_BCor_pre.sel(month=7)
      SSP3_pre_jul_df= SSP3_pre_jul.to_dataframe()
      SSP3_pre_jul_df = SSP3_pre_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jul_pre"})
      print('#######')
      SSP3_pre_aug = SSP3_BCor_pre.sel(month=8)
      SSP3_pre_aug_df= SSP3_pre_aug.to_dataframe()
      SSP3_pre_aug_df = SSP3_pre_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Aug_pre"})
      print('########')
      SSP3_pre_sep = SSP3_BCor_pre.sel(month=9)
      SSP3_pre_sep_df= SSP3_pre_sep.to_dataframe()
      SSP3_pre_sep_df = SSP3_pre_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Sep_pre"})
      print('#########')
      SSP3_pre_oct = SSP3_BCor_pre.sel(month=10)
      SSP3_pre_oct_df= SSP3_pre_oct.to_dataframe()
      SSP3_pre_oct_df = SSP3_pre_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Oct_pre"})
      print('##########')
      SSP3_pre_nov = SSP3_BCor_pre.sel(month=11)
      SSP3_pre_nov_df= SSP3_pre_nov.to_dataframe()
      SSP3_pre_nov_df = SSP3_pre_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Nov_pre"})
      print('###########')
      SSP3_pre_dec = SSP3_BCor_pre.sel(month=12)
      SSP3_pre_dec_df= SSP3_pre_dec.to_dataframe()
      SSP3_pre_dec_df = SSP3_pre_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Dec_pre"})
      print('############')
      SSP3_pre_df = pd.concat([SSP3_pre_jan_df, SSP3_pre_feb_df, SSP3_pre_mar_df, SSP3_pre_apr_df, SSP3_pre_may_df, SSP3_pre_jun_df, SSP3_pre_jul_df, SSP3_pre_aug_df, SSP3_pre_sep_df, SSP3_pre_oct_df, SSP3_pre_nov_df, SSP3_pre_dec_df], axis=1) # add each variable as a column
      SSP3_pre_df = SSP3_pre_df.reset_index() # add id column
      SSP3_pre_df.index = SSP3_pre_df.index + 1 # start id index at 1, not 0
      SSP3_pre_df.to_csv(pre_DIR+'bcor_SSP3.csv')
      print('SSP3_pre.csv complete')


# SSP5 pre
      SSP5_pre_jan = SSP5_BCor_pre.sel(month=1) # select pre data from the first month
      SSP5_pre_jan_df= SSP5_pre_jan.to_dataframe() # turn this data into a pandas dataframe
      SSP5_pre_jan_df = SSP5_pre_jan_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jan_pre"}) # drop unnecessary columns, rename variable columns to month
      print('#')
      SSP5_pre_feb = SSP5_BCor_pre.sel(month=2)
      SSP5_pre_feb_df= SSP5_pre_feb.to_dataframe()
      SSP5_pre_feb_df = SSP5_pre_feb_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Feb_pre"})
      print('##')
      SSP5_pre_mar = SSP5_BCor_pre.sel(month=3)
      SSP5_pre_mar_df= SSP5_pre_mar.to_dataframe()
      SSP5_pre_mar_df = SSP5_pre_mar_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Mar_pre"})
      print('###')
      SSP5_pre_apr = SSP5_BCor_pre.sel(month=4)
      SSP5_pre_apr_df= SSP5_pre_apr.to_dataframe()
      SSP5_pre_apr_df = SSP5_pre_apr_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Apr_pre"})
      print('####')
      SSP5_pre_may = SSP5_BCor_pre.sel(month=5)
      SSP5_pre_may_df= SSP5_pre_may.to_dataframe()
      SSP5_pre_may_df = SSP5_pre_may_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "May_pre"})
      print('#####')
      SSP5_pre_jun = SSP5_BCor_pre.sel(month=6)
      SSP5_pre_jun_df= SSP5_pre_jun.to_dataframe()
      SSP5_pre_jun_df = SSP5_pre_jun_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jun_pre"})
      print('######')
      SSP5_pre_jul = SSP5_BCor_pre.sel(month=7)
      SSP5_pre_jul_df= SSP5_pre_jul.to_dataframe()
      SSP5_pre_jul_df = SSP5_pre_jul_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Jul_pre"})
      print('#######')
      SSP5_pre_aug = SSP5_BCor_pre.sel(month=8)
      SSP5_pre_aug_df= SSP5_pre_aug.to_dataframe()
      SSP5_pre_aug_df = SSP5_pre_aug_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Aug_pre"})
      print('########')
      SSP5_pre_sep = SSP5_BCor_pre.sel(month=9)
      SSP5_pre_sep_df= SSP5_pre_sep.to_dataframe()
      SSP5_pre_sep_df = SSP5_pre_sep_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Sep_pre"})
      print('#########')
      SSP5_pre_oct = SSP5_BCor_pre.sel(month=10)
      SSP5_pre_oct_df= SSP5_pre_oct.to_dataframe()
      SSP5_pre_oct_df = SSP5_pre_oct_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Oct_pre"})
      print('##########')
      SSP5_pre_nov = SSP5_BCor_pre.sel(month=11)
      SSP5_pre_nov_df= SSP5_pre_nov.to_dataframe()
      SSP5_pre_nov_df = SSP5_pre_nov_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Nov_pre"})
      print('###########')
      SSP5_pre_dec = SSP5_BCor_pre.sel(month=12)
      SSP5_pre_dec_df= SSP5_pre_dec.to_dataframe()
      SSP5_pre_dec_df = SSP5_pre_dec_df.drop(["month"], axis=1).rename(columns={"mean monthly precipitation (mm)": "Dec_pre"})
      print('############')
      SSP5_pre_df = pd.concat([SSP5_pre_jan_df, SSP5_pre_feb_df, SSP5_pre_mar_df, SSP5_pre_apr_df, SSP5_pre_may_df, SSP5_pre_jun_df, SSP5_pre_jul_df, SSP5_pre_aug_df, SSP5_pre_sep_df, SSP5_pre_oct_df, SSP5_pre_nov_df, SSP5_pre_dec_df], axis=1) # add each variable as a column
      SSP5_pre_df = SSP5_pre_df.reset_index() # add id column
      SSP5_pre_df.index = SSP5_pre_df.index + 1 # start id index at 1, not 0
      SSP5_pre_df.to_csv(pre_DIR+'bcor_SSP5.csv')
      print('SSP5_pre.csv complete')

 # Turn warnings back on
      np.warnings.filterwarnings('default')
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
     
elif answer.lower().startswith("n"):
    pass
else:
        print("Enter either yes/no")

print('Step 8: Data Export complete')
################################################################################################################################
print('End.')
sys.exit()
