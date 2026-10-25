# rumz 2/12/25
# Processing code for hyperspectral ice core data

import os
import numpy as np
import matplotlib.pyplot as plt
import spectral

# Importing custom modules from the icespec package
from icespec import file_traverse_module as ft
from icespec import envi_capture_module as envi_capture
from icespec import preview_module as preview
from icespec import pca_module as pca
from icespec import equalizer_module as equalizer
from icespec import morphology_module as morph
from icespec import csv_capture_module as cap
from icespec import trim_module as trimmer
from icespec import light_correction_module as correction
from icespec import filter_bands_module as filter
from icespec import flatten_module as flat
from icespec import rematrix
from icespec import remove_bubbles_module as remove_bubbles
from icespec import channel_sum_module as sum_mod
# from icespec import reflectance_module as ref  # Currently not in use
# from icespec import preview_reflectance  # Currently not in use

spectral.settings.envi_support_nonlowercase_params = True  # Ensure compatibility with ENVI files

# Default RGB band indices for visualization
rd, gr, bl = 96, 59, 16


def getDataPath(dataPath: str, DataID: str, outputDir: str) -> str:
    """
    Retrieves the file path of the ENVI dataset corresponding to a given ice core sample.

    Args:
        dataPath (str): Path to the directory containing metadata.
        DataID (str): Identifier for the specific ice core sample.

    Returns:
        tuple: (datasetPath (str), top_depth (float), bottom_depth (float), core_length (float))
    """
    dataName, top_depth, bottom_depth, core_length, save_path, px_dims = cap.getData(DataID, dataPath, outputDir)
    datasetPath = os.path.join(os.path.dirname(dataPath), dataName)

    px_dims = [int(data.strip()) for data in px_dims.split(",")]

    print("Dataset Path: ", datasetPath)
    return datasetPath, top_depth, bottom_depth, core_length, save_path, px_dims

def getDenaliPath(scanNum: int, corePath: str) -> list:
    """
    Retrieves file paths for multiple scans of a given ice core sample. 
    ***This is in use until Denali scans are added to HSIJuly.csv***

    Args:
        scanNum (int): The number of scans to retrieve.
        corePath (str): The directory containing the core scans.

    Returns:
        list: Paths to the selected ENVI scan files.
    """
    datasetPaths = ft.getScans(scanNum, corePath)
    return datasetPaths

def getEnviData(data: str, Debug=False):
    """
    Loads hyperspectral ENVI data along with white and dark reference images.

    Args:
        data (str): Path to the ENVI dataset.
        Debug (bool, optional): Enables additional previewing if True.

    Returns:
        tuple: (data_nparr (np.ndarray), w_arr (np.ndarray), d_arr (np.ndarray), fwhm (np.ndarray))
    """
    data_nparr, w_arr, d_arr, fwhm = envi_capture.capture(data, Debug)
    preview.interactive_preview(data_nparr, "Raw data")
    return data_nparr, w_arr, d_arr, fwhm


def trim(data_nparr: np.ndarray, w_ref: np.ndarray, d_ref: np.ndarray, dataSize: list, w_stdSize: list, Debug: bool):
    """
    Crops hyperspectral images to a specified width and height.

    Args:
        data_nparr (np.ndarray): Hyperspectral image array.
        w_ref (np.ndarray): White reference image.
        d_ref (np.ndarray): Dark reference image.
        dataSize (list): Desired dimensions for data array.
        w_stdSize (list): Standard dimensions for white reference.
        Debug (bool): Enables additional previews if True.

    Returns:
        list: Trimmed data arrays.
    """
    trimmedData = trimmer.trim(data_nparr, w_ref, d_ref, dataSize, w_stdSize, Debug)
    preview.show_rgb(trimmedData[0], "Trimmed Raw Data")
    
    if Debug:
        preview.show_rgb(trimmedData[1], "Trimmed White Reference")

    return trimmedData

def extract_sections(trimmedData, section_width=50):
    image = trimmedData[0]
    w_ref = trimmedData[1]
    d_ref = trimmedData[2]
    height, width, bands = image.shape

    # Ensure section width doesn't exceed image size
    section_width = min(section_width, width // 3)

    # Calculate start points for left, middle, and right
    left_start = 0
    middle_start = (width // 2) - (section_width // 2)
    right_start = width - section_width

    # Extract sections
    left_section = image[:, left_start:left_start + section_width, :]
    left_wref = w_ref[:, left_start:left_start + section_width, :]
    left_dref = d_ref[:, left_start:left_start + section_width, :]
    middle_section = image[:, middle_start:middle_start + section_width, :]
    middle_wref = w_ref[:, middle_start:middle_start + section_width, :]
    middle_dref = d_ref[:, middle_start:middle_start + section_width, :]
    right_section = image[:, right_start:right_start + section_width, :]
    right_wref = w_ref[:, right_start:right_start + section_width, :]
    right_dref = d_ref[:, right_start:right_start + section_width, :]

    left = [left_section, left_wref, left_dref]
    middle = [middle_section, middle_wref, middle_dref]
    right = [right_section, right_wref, right_dref]
    

    return left, middle, right

def reflectance(trimmedData: list, Debug: bool):
    """
    Corrects reflectance errors in hyperspectral images.

    Args:
        trimmedData (list): List of trimmed hyperspectral arrays.
        Debug (bool): Enables additional debug output.

    Returns:
        np.ndarray: Reflectance-corrected image.
    """
    corrected_img = correction.correct(trimmedData)
    
    if Debug:
        print(f"Reflectance calculated. Dimensions: {corrected_img.shape}")

    return corrected_img

def remove_bands(corrected_img, w_ref):
    """
    Removes spectral bands that provide little useful information.

    Args:
        corrected_img (np.ndarray): Reflectance-corrected image.
        w_ref (np.ndarray): White reference image used for filtering.

    Returns:
        np.ndarray: Image with irrelevant spectral bands removed.
    """
    filtered_img = filter.filter(corrected_img, w_ref)
    return filtered_img

def previewImg(hsi_data, dsetName,  bands=(1, 6, 29), figsize=(24, 36)):
    """
    Displays an RGB preview of hyperspectral image data.

    Args:
        hsi_data (np.ndarray): Hyperspectral image data.
        dsetName (str): Name of the dataset being previewed.
        bands (tuple, optional): Indices of the three spectral bands to use for RGB visualization.
        figsize (tuple, optional): Figure size for display.

    Returns:
        int: Always returns 0 (no functional return value).
    """
    rgb_img = [rd, gr, bl]
    preview.show_rgb(hsi_data, dsetName, bands, figsize)
    return 0

def bubbles(img, fwhm):
    """
    Detects and removes bubbles in hyperspectral ice core images.

    Args:
        img (np.ndarray): Hyperspectral image data.
        fwhm (np.ndarray): Full-width at half-maximum values for spectral bands.

    Returns:
        np.ndarray: Image with bubble pixels removed and inpainted.
    """
    inpainted_data = remove_bubbles.remove(img, fwhm)
    return inpainted_data

def flatten_data(filtered_img):
    """
    Flattens a 3D hyperspectral image into a 2D array for processing.

    Args:
        filtered_img (np.ndarray): Hyperspectral image with irrelevant bands removed.

    Returns:
        np.ndarray: 2D array representation of the image.
    """
    flattened_img = flat.flatten(filtered_img)
    return flattened_img

def resize_matrix(flattened_img, trimmedData, pc_num=5):
    """
    Reshapes a flattened hyperspectral image into a 3D array with principal components.

    Args:
        flattened_img (np.ndarray): 2D representation of the hyperspectral image.
        trimmedData (list): Original trimmed data for reference.
        pc_num (int, optional): Number of principal components to retain. Default is 5.

    Returns:
        np.ndarray: 3D array with reshaped principal components.
    """
    rematrix_img = rematrix.new_matrix(flattened_img, trimmedData, pc_num)
    print("Rematrix shape: ", rematrix_img.shape)
    preview.show_pca_rgb(rematrix_img, "Rematrixed Data")
    return rematrix_img

def equalize(image, save, savePath):
    """
    Enhances image contrast using histogram equalization.

    Args:
        image (np.ndarray): Input image data.

    Returns:
        np.ndarray: Equalized image.
    """
    image.astype(np.uint8)  # Convert to 8-bit format before equalization
    equalized, img_localeq = equalizer.analyze_data(image, save, savePath)
    return img_localeq

def channel_sums(img: np.ndarray, Debug=False, save=False, savePath="") -> np.ndarray:
    """
    Gets sum of each line per channel, then uses channel sums to remake image.

    Args:
     - img: (np.ndarray): Reflectance-corrected and bubble filled image.
     - Debug (bool, optional): If True, prints debug information about bubble counts.
     - save (bool, optional): If True, saves image to specified folder.
     - savePath (str, optional): used if save is True, specifes path to save folder and file name.

     Returns:
        np.ndarray: Sum Value image.
    """
    sums_img = sum_mod.get_sums(img, Debug, save, savePath)

    return sums_img



def components(corrected_img: np.ndarray, pc_num):
    """
    Extracts principal components from hyperspectral images.
    (Not currently used, replaced by rematrix method.)

    Args:
        corrected_img (np.ndarray): Reflectance-corrected image.
        pc_num (int): Number of principal components to retain.

    Returns:
        np.ndarray: Image transformed using principal component analysis.
    """
    pca_img = pca.variance(corrected_img, pc_num)
    return pca_img

def morphin(eq_img):
    """
    Applies morphological processing to an image.
    (Not currently in use.)

    Args:
        eq_img (np.ndarray): Equalized image.

    Returns:
        np.ndarray: Morphologically processed image.
    """
    morphed_img = morph.process(eq_img)
    return morphed_img


# Unused function for data preview (commented out)
# def preview(img, corrected_img):
#     preview_reflectance.data_preview(img, corrected_img)
#     return 0