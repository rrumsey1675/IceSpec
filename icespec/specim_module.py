"""This module reads SPECIM envi files. 

"""
#import libraries

from spectral import *
from spectral import imshow
import spectral.io.envi as envi
spectral.settings.envi_support_nonlowercase_params = True
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt


def read_envi(dsetName:str='SPC-14-1317', path2file: str ='/image',imageDIR: str='capture') -> np.array:
    """Read envi files.

    Args:
        lst (List[int]): Some list of non-None integers

    Returns:
        np.array: with image data
    """
    os.environ['SPECTRAL_DATA'] = f"{path2file}/{imageDIR}"
    data_captured = envi.open(dsetName+'.hdr')
    result = np.array(data_captured.load())   
    return result

    #Add code to capture exceptions
def hsi_show(hsidata:np.array, dsetName: str='SPC-14-1317', rgb_ch: List[int]=[16,59,96],figsize: List[int]=[24,36])-> ImageView:
    v=imshow(hsidata,rgb_ch,stretch_all=True, title=dsetName,  figsize=figsize)
    return v # returns ImageView object (can be printed)  
################################
def trim_raw(raw_data, edge: List[int]=[0,0,0,0])-> np.array:
    """
    Takes a full hyperspectral data and crops margins.
    
    Args:
    
    - full_data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to be cropped (Lines, Samples and Bands)
    - removal in pixels as int value top, bottom, left, right.

    Returns:
           data_cropped (3D np.ndarray Float 32): 3D array representing the full cropped hyperspectral data (LInes, Samples and Bands)
    """
    
    lines, samples, bands = raw_data.shape  #Gets the shape of the data, saving it to the hight width and band variables.

    top_margin = edge[0]
    bottom_margin = edge[1]
    left_margin = edge[2]
    right_margin = edge[3]    
    data_cropped=raw_data[top_margin:bottom_margin, left_margin:samples-right_margin, :] #Crops the representative band and full data by the calculated margins
    return data_cropped # Return the  cropped data
    
    
#################################################
def layer_calc_med(data)-> np.array:
    """
    Takes  hyperspectral data and calculates median value across line of the image for all bands
    
    Args:
    
    - data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to (Lines, Samples and Bands)

    Returns:
           median_data (2D np.ndarray Float 32): 2D array representing median value of all samples per line for every band
    """
    bands=data.shape[2]
    lines_med=data.shape[0]
    #initialize array with zeros with dimensions we need
    data_med = np.zeros((lines_med,bands))
    for n in range(bands):
        data_one_band=data[:,:,n]
        data_med[:,n]=np.median(data_one_band, axis=1)
    return data_med
    
#################################################
def layer_calc_sum(data)-> np.array:
    """
    Takes  hyperspectral data and calculates sum of values across line of the image for all bands
    
    Args:
    
    - data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to (Lines, Samples and Bands)

    Returns:
         sum_data (2D np.ndarray Float 32): 2D array representing median value of all samples per line for every band
    """
    bands=data.shape[2]
    lines_sum=data.shape[0]
    #initialize array with zeros with dimensions we need
    data_sum = np.zeros((lines_sum,bands))
    for n in range(bands):
        data_one_band=data[:,:,n]
        data_sum[:,n]=np.sum(data_one_band, axis=1)
    return data_sum
#################################################
def signal_var(data)-> np.array:
    
    """
    Takes  trimmed hyperspectral data and calculates 
    
    Args:
    
    - data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to (Lines, Samples and Bands)
    
    Returns:
         var_perChannel  (2D np.ndarray Float 32): 2D array representing variability value per channel
    """
    bands= data.shape[2] #calculate how many bands
    data_var = []
    for n in range(bands):
        data_one_band=data[:,:,n].flatten()
        data_var.append(np.var(data_one_band))
    return data_var
#################################################
def signal_mean(data)-> np.array:
    
    """
    Takes  trimmed hyperspectral data and calculates mean
    
    Args:
    
    - data (3D np.ndarray Float 32): 3D array containing all the hyperspectral data to (Lines, Samples and Bands)
    
    Returns:
         var_perChannel  (2D np.ndarray Float 32): 2D array representing variability value per channel
    """
    bands= data.shape[2] #calculate how many bands
    data_mean = []
    for n in range(bands):
        data_one_band=data[:,:,n].flatten()
        data_var.append(np.mean(data_one_band))

    return data_mean