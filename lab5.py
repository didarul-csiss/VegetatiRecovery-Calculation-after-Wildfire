### Part 1

import rasterio
import os
import numpy as np
import pandas as pd
%matplotlib inline

# Data dir
path = "C:/Users/didar/Downloads/data_1/data/"
fp = os.path.join(path, "bigElk_dem.tif")

# Open dem the file:
dem = rasterio.open(fp)


"""
Author: Originally created by Galen Maclaurin, updated by Ricardo Oliveira
Created: Created on 3.15.16, updated on 10.17.19
Purpose: Helper functions to get started with Lab 5
"""

import numpy as np


def slopeAspect(dem, cs):
    """Calculates slope and aspect using the 3rd-order finite difference method

    Parameters
    ----------
    dem : numpy array
        A numpy array of a DEM
    cs : float
        The cell size of the original DEM

    Returns
    -------
    numpy arrays
        Slope and Aspect arrays
    """

    from math import pi
    from scipy import ndimage
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    dzdx = ndimage.convolve(dem, kernel, mode='mirror') / (8 * cs)
    dzdy = ndimage.convolve(dem, kernel.T, mode='mirror') / (8 * cs)
    slp = np.arctan((dzdx ** 2 + dzdy ** 2) ** 0.5) * 180 / pi
    ang = np.arctan2(-dzdy, dzdx) * 180 / pi
    aspect = np.where(ang > 90, 450 - ang, 90 - ang)
    return slp, aspect


def reclassAspect(npArray):
    """Reclassify aspect array to 8 cardinal directions (N,NE,E,SE,S,SW,W,NW),
    encoded 1 to 8, respectively (same as ArcGIS aspect classes).

    Parameters
    ----------
    npArray : numpy array
        numpy array with aspect values 0 to 360

    Returns
    -------
    numpy array
        numpy array with cardinal directions
    """
    return np.where((npArray > 22.5) & (npArray <= 67.5), 2,
    np.where((npArray > 67.5) & (npArray <= 112.5), 3,
    np.where((npArray > 112.5) & (npArray <= 157.5), 4,
    np.where((npArray > 157.5) & (npArray <= 202.5), 5,
    np.where((npArray > 202.5) & (npArray <= 247.5), 6,
    np.where((npArray > 247.5) & (npArray <= 292.5), 7,
    np.where((npArray > 292.5) & (npArray <= 337.5), 8, 1)))))))


def reclassByHisto(npArray, bins):
    """Reclassify np array based on a histogram approach using a specified
    number of bins. Returns the reclassified numpy array and the classes from
    the histogram.

    Parameters
    ----------
    npArray : numpy array
        Array to be reclassified
    bins : int
        Number of bins

    Returns
    -------
    numpy array
        umpy array with reclassified values
    """
    # array = np.where(np.isnan(npArray), 0, npArray)
    histo = np.histogram(~np.isnan(npArray), bins)[1]
    rClss = np.zeros_like(npArray)
    for i in range(bins):
        #print(i + 1, histo[i], histo[i + 1])
        #print(np.where((npArray > histo[i]) & (npArray <= histo[i + 1])))
        rClss = np.where((npArray >= histo[i]) & (npArray <= histo[i + 1]),
                         i + 1, rClss)
    return rClss

# processing dem
def dem_process(img):
    cellX=img.res
    slope, aspect=slopeAspect(img, cellX)
    reclass=reclassAspect(slope)

# NDVI calculation    
def ndvi(red,nri):
    red1 = rasterio.open(red)
    red_=red1.read(1)
    nri = rasterio.open(nri)
    nri_=nri.read(1)
    return (nri_-red_)/(nri_+red_)    

# polyfit
def polyfit_(df,columns):
    coeff_list=[]
    for i in columns:
        data=df[i]
        coeff = np.polyfit(data,data.index,1)
        coeff_list.append(coeff[0])
    return coeff_list


# DEMcell size
cellX, cellY  =dem.res

# Generating slope and aspect from DEM
dem_=dem.read(1)
slope, aspect=slopeAspect(dem_,cellX)

# Reclassifying slope and aspect
aspect_reclass=reclassAspect(aspect)
slope_reclass=reclassByHisto(slope, 10)

# Loading all bands and seperating Band3 and Band4
import glob
bands=glob.glob(path+'L5_big_elk/*'+'.tif')

# creating list to hold b3 and b4
b3=[]
b4=[]
for i in bands:
    if i[-6:-4]=='B3':  # matching B3 and B4 in the file name
        b3.append(i)
    else:
        b4.append(i)

# calculating NDVI for all years and holding them in a list
ndvi_list=[]
for a,b in zip(b3,b4):
    ndvi_=ndvi(a,b)
    ndvi_list.append(ndvi_)

# Loading fire perimeter data
fire_per=rasterio.open(path+'fire_perimeter.tif')
forest=fire_per.read(1)
# healthy forest
healthy_forest_=forest==2
# converting boolean values to numeric 0 and 1
healthy_forest_num=healthy_forest_*1

# calculate mean NDVI
mean_ndvi=[]
for i in ndvi_list:
    mean_ndvi.append(np.mean(i[forest==2]))
print("Mean NDVI for years 2002-2011",mean_ndvi)

# RR calculation for all ndvi
rr=[]
for i,j in zip(ndvi_list,mean_ndvi):
    rr.append(i/j)

# flattening rr array
rr_flat=[]
for i in rr:
    arr=np.ravel(i)
    rr_flat.append(arr)
# creating a dataframe for convenience in polyfit function implementation
df=pd.DataFrame(np.array(rr_flat))
# polyfit for all columns (each column represent a year, 2002-2011)
coeff=polyfit_(df,df.columns)
print("DEM Shape",np.shape(dem))
# reshapeing flat coefficient to 2d array
reshape_coeff=np.reshape(coeff, (280,459))

# print mean coef  only for burned forest areas
print("Mean coeff for all years for burned forest",np.mean(reshape_coeff[healthy_forest==1]))

# print mean rr for healty forest
for i in rr:
    area_mask=i[forest==2]
    print("Yearly Mean RR for healty forest",np.mean(area_mask))

# print mean rr for healty forest
for i in rr:
    print("Yearly Mean RR",np.mean(i))




### Part 2

# defining funciton for zonal statistics
def min(coef,cls):
    min_=[]
    cls=cls[forest==1] # masking burned arras
    coef=coef[forest==1] # masking burned forest arras
    for i in np.unique(cls):
        mask=coef[cls==i]
        min_.append(np.min(mask))
    return min_

def max(coef,cls):
    max_=[]
    cls=cls[forest==1]
    coef=coef[forest==1]
    for i in np.unique(cls):
        mask=coef[cls==i]
        max_.append(np.max(mask))
    return max_
               
def mean(coef,cls):
    mean_=[]
    cls=cls[forest==1]
    coef=coef[forest==1]
    for i in np.unique(cls):
        mask=coef[cls==i]
        mean_.append(np.mean(mask))
    return mean_
               
def std(coef,cls):
    std_=[]
    cls=cls[forest==1]
    coef=coef[forest==1]
    for i in np.unique(cls):
        mask=coef[cls==i]
        std_.append(np.std(mask))
    return std_    

def count(coef,cls):
    count_=[]
    cls=cls[forest==1]
    coef=coef[forest==1]
    for i in np.unique(cls):
        mask=coef[cls==i]
        count_.append(len(np.ravel(mask)))
    return count_

# Creating zonal statistics dataframe for aspect
aspect_ = pd.DataFrame(list(zip(np.unique(aspect_reclass), min(reshape_coeff,aspect_reclass),max(reshape_coeff,aspect_reclass),mean(reshape_coeff,aspect_reclass),
                          std(reshape_coeff,aspect_reclass), count(reshape_coeff,aspect_reclass))),
               columns =['aspect_class', 'min','max','mean','std','count'])
aspect_.to_csv(path+'aspect_coeff.csv',index=False)

# Creating zonal statistics dataframe for slope
slope_ = pd.DataFrame(list(zip(np.unique(slope_reclass), min(reshape_coeff,slope_reclass),max(reshape_coeff,slope_reclass),mean(reshape_coeff,slope_reclass),
                          std(reshape_coeff,slope_reclass), count(reshape_coeff,slope_reclass))),
               columns =['slope_reclass', 'min','max','mean','std','count'])
slope_.to_csv(path+'slope_coeff.csv',index=False)

# creating coeff raster and masking out burned areas -99
mask_coeff=reshape_coeff*healthy_forest_
mask_coeff[mask_coeff == 0.] = -99

# opening dem raster as reference
with rasterio.open(path+'bigElk_dem.tif') as src:
    ras_meta = src.profile
# saving coefficient as raster
with rasterio.open(path+'coeff_raster.tif', 'w', **ras_meta) as dst:
    dst.write(mask_coeff, indexes=1)
