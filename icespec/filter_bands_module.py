# Get noise ratio, drop bands below threshold?
# code from Noise Test

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import spectral
from typing import List
import numpy as np
from scipy.signal import convolve2d
import math
from skimage.restoration import estimate_sigma




import numpy as np
import math
from scipy.signal import convolve2d

def filter(hsi_data: np.array, wht_data: np.array, wvls, fwhm: np.array, Debug=False, outputPath="", save=False) -> np.array:
    """
    Removes spectral bands based on the SNR Proxy: sigma_whiteref / Noise_data
    """

    # Step 1: Remove vertical artifacts
    # for band in range(hsi_data.shape[2]):
    #     hsi_data[:, :, band] = remove_vertical_artifact(hsi_data[:, :, band])

    # Step 2: Noise filtering and SNR Proxy Calculation
    bands = hsi_data.shape[2]
    SNR_list = []
    
    noise_data = [] # To store Immerkaer noise 
    sigma_whiteref = [] # To store Std Dev of white ref

    for n in range(bands):
        # 1. Calculate Noise_data using Immerkaer method on ice core data
        noise_val = estimate_noise(hsi_data[:, :, n])
        noise_data.append(noise_val)

        # 2. Calculate sigma_whiteref (Standard Deviation) of the white reference
        wht_band_flat = wht_data[:, :, n].flatten()
        sigma_wht_val = np.std(wht_band_flat)
        sigma_whiteref.append(sigma_wht_val)

        # 3. SNR Proxy Calculation: sigma_whiteref / Noise_data
        # (Added 1e-10 to prevent division by zero)
        proxy_val = sigma_wht_val / (noise_val + 1e-10)
        SNR_list.append(proxy_val)

    SNR = np.array(SNR_list)


    # --- DIAGNOSTIC PRINTS ---
    if Debug:
        print(f"--- SNR Proxy Stats ---")
        print(f"Min SNR: {np.min(SNR):.4f}")
        print(f"Max SNR: {np.max(SNR):.4f}")
        print(f"Mean SNR: {np.mean(SNR):.4f}")
        print(f"-----------------------")
        print(SNR)
        print(f"-----------------------")

    # Convert to numpy arrays 
    noise_data = np.array(noise_data)
    sigma_whiteref = np.array(sigma_whiteref)

    # Step 3: Apply formula:
    # SNR_proxy = sigma_whiteref / Noise_data
    # add tiny epsilon (1e-10) to avoid division by zero
    SNR = sigma_whiteref / (noise_data + 1e-10)

    plt.clf()
    plt.figure(figsize=(10, 10))
    y = np.arange(1, bands+1)
    plt.title("SNR per channel")
    #plt.axis([0, 223, 0, 7])
    plt.xlabel("Channel: 1-224") 
    plt.ylabel("SNR")
    plt.plot(y, SNR)
    out_path = Path(outputPath) / "SNR-graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Apply thresholding
    hsi_data, wvls, fwhm = threshold(SNR, hsi_data, wvls, fwhm, Debug, outputPath, save)
    
    print("Shape After: ", hsi_data.shape)
    
    return hsi_data, wvls, fwhm

def threshold(SNR, hsi_data, wvls, fwhm, Debug=False, outputPath="", save=False):
    # function for thresholding noise and removing bad bands
    """
    Calculates threshold and cuts useless bands

    Args:
        SNR (np.ndarray): Spectral Noise Ratio
        hsi_data (np.ndarray): hsi image array

    Returns:
        np.ndarray: Image with irrelevant spectral bands removed.
    """
    good_bands = []
    bad_bands = []

    for n in range(0, 224):
        if SNR[n] <= 3.5 or SNR[n] > 18 or n < 10 or 180 < n < 210:
            bad_bands.append(n)
        else:
            good_bands.append(n)
            
    if Debug:
        if save:
            readme_path = outputPath / "README.md"
            with readme_path.open("a") as f:
                f.write(f"\n###### DEBUG FILTER\n")
                f.write(f"\n### Good Bands:\n- {good_bands}\n")
                f.write(f"### Bad Bands:\n- {bad_bands}\n")
                f.write(f"### Shape Before: {hsi_data.shape}\n")
        print("Good Bands: ", good_bands)
        print("Bad Bands: ", bad_bands)
        print("Shape Before: ", hsi_data.shape)
        
    if good_bands != []:
        hsi_data = np.delete(hsi_data, bad_bands, axis=2)
        wvls = np.delete(wvls, bad_bands)
        fwhm = np.delete(fwhm, bad_bands)
        if Debug:
            print("Good WVs: ", wvls)

    if good_bands == []:
        bad_bands = []
        for n in range(0, 224):
            # if SNR[n] <= 3.5 or SNR[n] > 11080 or n < 10 or 180 < n < 210:
            if SNR[n] <= 3000 or SNR[n] > 12000 or n < 10 or n > 210:
                bad_bands.append(n)
            else:
                good_bands.append(n)

        hsi_data = np.delete(hsi_data, bad_bands, axis=2)
        wvls = np.delete(wvls, bad_bands)
        fwhm = np.delete(fwhm, bad_bands)
        print("Take 2: Good WVs: ", wvls)
        

    return hsi_data, wvls, fwhm

def remove_vertical_artifact(image:np.ndarray, threshold:float=1.5) -> np.ndarray :
    """
    Removes vertical striping caused by planing/cutting.

    Args:
    - image (np.ndarray): 2D grayscale image data.
    - threshold (float): Intensity deviation threshold to identify stripe regions.

    Returns:
    - corrected_image (np.ndarray): Image with vertical stripes suppressed.
    """
    corrected_image = image.copy()
    # Calculate the column-wise mean
    column_means = np.mean(image, axis=0)
    global_mean = np.mean(column_means)
    deviations = np.abs(column_means - global_mean)

    # Identify vertical striping regions (columns with deviation above threshold)
    artifact_columns = np.where(deviations > threshold)[0]

    for col in artifact_columns:
        corrected_image[:, col] = np.mean(corrected_image[:, max(0, col - 1):col + 2], axis=1)

    return corrected_image


def estimate_noise(I):
  """
    Estimate noise for a given spectral band.

    Args:
        I (np.ndarray): one band from hsi_data image.

    Returns:
        sigma: estimated noise for the given band.
    """

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma



def estimate_noise_wavelet(img):
    return estimate_sigma(img)

def showGraphs(data_var, wht_var, x):
    # shows noise ratio graphs
    plt.title("SNR per channel")
    plt.xlabel("Channel: 1-224") 
    plt.ylabel("SNR")
    plt.plot(x, np.sqrt(np.array(data_var)**2 / np.array(wht_var)**2 - 1))
    # plt.plot(x, np.sqrt(np.array(data_var)**2 / np.array(bkg_var)**2 - 1))
    plt.show()