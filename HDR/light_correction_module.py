import pandas as pd
import numpy as np
import spectral
spectral.settings.envi_support_nonlowercase_params = True
import spectral.io.envi as envi

# corrected_data = np.divide(
#             np.subtract(img_data, d_ref),
#             np.subtract(w_ref, d_ref))

# add a col size check

def correct(trimmedData: list) -> np.array: #trimmedData is a list of Envi files, raw data, white reference, and dark reference
    """
    Corrects reflectance errors in hyperspectral images.

    Args:
        trimmedData (list): List of trimmed hyperspectral arrays.
        Debug (bool): Enables additional debug output.

    Returns:
        np.ndarray: Reflectance-corrected image.
    """

    img_data = trimmedData[0]
    w_ref = trimmedData[1]
    d_ref = trimmedData[2]

    rows = img_data.shape[0] #lines scanned
    samples = img_data.shape[1] #samples 1-1024 max 
    bands = img_data.shape[2] # also channels 1-224

    #Calculate mean value per channel for dark reference
    d_ref_mean=d_ref.mean(axis=0)
    #Calculate mean value per channel for white reference
    w_ref_mean=w_ref.mean(axis=0)


    denominator = w_ref_mean - d_ref_mean
    # print(f"W ref: {w_ref_mean.shape}, D ref: {d_ref_mean.shape}, Denom:  {denominator.shape}")

    #beforeArrs for checking/debugging
    before1Arr = img_data[50,250:350,0]
    before2Arr = img_data[50,250:350,99]
    before3Arr = img_data[50,250:350,223]
    
    ref_img = np.empty((bands, rows, samples))
    
    #Transpose data to speed up operations
    img_data=np.transpose(img_data, (2, 0, 1))
    denominator = denominator.T
    d_ref_mean = d_ref_mean.T

    for i, im_ch in enumerate(img_data):  
        denom = denominator[i, :]   
        dark = d_ref_mean[i, :]    
        valid_mask = denom != 0 
        corrected_band = np.zeros_like(im_ch, dtype=float)
        # Apply correction only on valid columns
        corrected_band[:, valid_mask] = (im_ch[:, valid_mask] - dark[valid_mask]) / denom[valid_mask]
        # Fill invalid columns with mean
        band_mean = np.mean(corrected_band[:, valid_mask])
        corrected_band[:, ~valid_mask] = band_mean
        ref_img[i] = corrected_band

    # print("Reflectance image transposed: ", ref_img.shape)

    #Transpose image back to the original dimensions
    ref_img=np.transpose(ref_img, (1, 2, 0))

    #afterArrs for checking/debugging
    after1Arr = ref_img[50,250:350,0]
    after2Arr = ref_img[50,250:350,99]
    after3Arr = ref_img[50,250:350,223]

   
    
    return ref_img