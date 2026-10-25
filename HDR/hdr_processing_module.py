# rumz 4/24/25
# Processing code for hyperspectral hdr

import os
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import spectral
from pathlib import Path

# Importing custom modules from the icespec package

from HDR import hdr_envi_capture_module as envi_capture
from HDR import hdr_csv_capture_module as cap
from HDR import DENhdr_csv_capture_module as DENcap
from HDR import HDRmk_output as mkpath
from HDR import hdr_auto_trim_module as trimmer
from HDR import light_correction_module as correction
from HDR import hdr_module as hdr
from HDR import preview_module as preview



spectral.settings.envi_support_nonlowercase_params = True  # Ensure compatibility with ENVI files

# Default RGB band indices for visualization
rd, gr, bl = 96, 59, 16


def getDataPath(CoreID: str, SectionID: str, csv_path: str, output_dir: str, Debug=False) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Retrieves metadata for a given ice core section, and selects the correct images for:
        - reflectance imaging
        - transmitted imaging (high and low exposure)
    And then prints the metadata for user.

    Args:
        CoreID (str): Identifier for the specific ice core (e.g., 'ALHIC2302').
        SectionID (str): Section ID within the core (e.g., '30 Sec 34', '2404').
        csv_path (str): Path to the CSV file containing scan metadata.
        output_dir (str): Path to the base directory where outputs will be saved.

    Returns:
        Tuple of 3 coreData lists:
            - reflectance_data: Corresponds to the reflectance image
            - transmitted_lo_data: Corresponds to the lower-exposure transmitted image
            - transmitted_hi_data: Corresponds to the higher-exposure transmitted image

        Each coreData is a list with:
            [0] dataPath (str): Full path to the ENVI scan directory
            [1] top_depth (float)
            [2] bottom_depth (float)
            [3] core_length (float)
            [4] save_path (str)
            [5] px_dims (List[int])
            [6] topLight (int)
            [7] bottomLight (int)
            [8] exposure (int)
    """

    [final_reflectance, transmitted_lo, transmitted_hi] = cap.getData(CoreID, SectionID, csv_path, output_dir)
    
    if Debug:
        print_core_image_info(final_reflectance, transmitted_lo, transmitted_hi)

    return [final_reflectance, transmitted_lo, transmitted_hi]

def getDENPath(data_ids: List[str], csv_path: str, output_dir: str, Debug=False) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Retrieves metadata for a given ice core section, and selects the correct images for:
        - reflectance imaging
        - transmitted imaging (high and low exposure)
    And then prints the metadata for user.

    Args:
        data_ids List(str): Identifiers for the specific hsi images.
        csv_path (str): Path to the CSV file containing scan metadata.
        output_dir (str): Path to the base directory where outputs will be saved.

    Returns:
        Tuple of 3 coreData lists:
            - reflectance_data: Corresponds to the reflectance image
            - transmitted_lo_data: Corresponds to the lower-exposure transmitted image
            - transmitted_hi_data: Corresponds to the higher-exposure transmitted image

        Each coreData is a list with:
            [0] dataPath (str): Full path to the ENVI scan directory
            [1] top_depth (float)
            [2] bottom_depth (float)
            [3] core_length (float)
            [4] save_path (str)
            [5] px_dims (List[int])
            [6] topLight (int)
            [7] bottomLight (int)
            [8] exposure (int)
    """

    [final_reflectance, transmitted_lo, transmitted_hi] = DENcap.getData(data_ids, csv_path, output_dir)
    
    if Debug:
        print_core_image_info(final_reflectance, transmitted_lo, transmitted_hi)

    return [final_reflectance, transmitted_lo, transmitted_hi]

def make_dir(dataList: Tuple[List[Any], List[Any], List[Any]], script_id: str, notes: str = "", batch_id: str = "") -> Path:

    savePath = mkpath.OutputPath(dataList, script_id, notes, batch_id)

    return savePath
    
def getEnviData(dataList: Tuple[List[Any], List[Any], List[Any]], Debug: bool = False) -> List[List[np.ndarray]]:
    """
    Loads hyperspectral ENVI data along with white and dark reference images for three images:
    - Reflectance
    - Transmitted low exposure
    - Transmitted high exposure

    Args:
        dataList (tuple): A tuple containing three coreData lists:
            - [0]: final_reflectance
            - [1]: transmitted_lo
            - [2]: transmitted_hi
        Debug (bool, optional): If True, enables additional debug output.

    Returns:
        List[List[np.ndarray]]: A list of three sublists, each containing:
            [data_nparr, w_arr, d_arr, fwhm]
    """
    imgs = []
    
    for coreData in dataList:
        data_path = coreData[0]  # dataPath is at index 0
        data_nparr, w_arr, d_arr, fwhm, wvls = envi_capture.capture(data_path, Debug)
        imgs.append([data_nparr, w_arr, d_arr, fwhm, wvls])
        
    return imgs

def trim(imgs: List[List[np.ndarray]], dataList: List[List], stdSize: List[int] = [530, 820, 200, 900], Debug: bool = False) -> List[List[np.ndarray]]:
    """
     Trims hyperspectral image sets (raw, white ref, dark ref, fwhm) based on standard size.

    Args:
        imgs (List[List[np.ndarray]]): A list where each element is a list of 4 arrays:
            [0] raw image (np.ndarray)
            [1] white reference (np.ndarray)
            [2] dark reference (np.ndarray)
            [3] fwhm array (np.ndarray)
        dataList (List[List]): List of core metadata (used to extract px_dims).
        stdSize (List[int], optional): Standard size [height, width, bands, max_bands].
        Debug (bool, optional): If True, enables debug visualization.

    Returns:
        List[List[np.ndarray]]: Trimmed data arrays for each image, in the same format.
    """
    dataSize = dataList[0][5] #px_dims
    # dataSize = [0, 4960, 0, 1024]
    #dataSize = [1373, 5115, 14,560]
    trimmedArrs = []

    for img in imgs:
        trimmedData = trimmer.trim(img[0], img[1], img[2], dataSize, stdSize, Debug)
        preview.show_rgb(trimmedData[0], "Trimmed Raw Data")
        trimmedArrs.append(trimmedData)
    
        if Debug:
            preview.show_rgb(trimmedData[1], "Trimmed White Reference")

    return trimmedArrs

def reflectance(trimmedArrs: List[List[np.ndarray]], Debug: bool = False) -> List[np.ndarray]:
    """
    Applies reflectance correction to each trimmed hyperspectral cube set.

    Args:
        trimmedData (List[List[np.ndarray]]): Each inner list contains:
            [0] raw image
            [1] white reference
            [2] dark reference
            [3] fwhm (optional or retained)
        Debug (bool): If True, prints out shape of corrected output.

    Returns:
        List[np.ndarray]: Reflectance-corrected image cubes.
    """
    corrected_imgs = []
    
    for core in trimmedArrs:
        corrected_img = correction.correct(core)
        corrected_imgs.append(corrected_img)
    
    if Debug and corrected_imgs:
        print(f"Reflectance calculated. Dimensions: {corrected_imgs[0].shape}\n{corrected_imgs[1].shape}\n{corrected_imgs[2].shape}")

    return corrected_imgs

def combine_imgs(corrected_imgs, Debug, save, savepath):

    hdr_img = hdr.combine_hsi(corrected_imgs[0], corrected_imgs[1], corrected_imgs[2], Debug, save=save, savepath=savepath)

    return hdr_img

def print_core_image_info(reflectance: List, transmitted_lo: List, transmitted_hi: List) -> None:
    """
    Nicely formats and prints metadata for three hyperspectral images:
    - One reflectance image
    - One low-exposure transmitted image
    - One high-exposure transmitted image

    Each image's metadata is expected to follow the standard `coreData` structure:
        [0] dataPath (str): Path to the ENVI data directory
        [1] top_depth (float)
        [2] bottom_depth (float)
        [3] core_length (float)
        [4] save_path (str)
        [5] px_dims (List[int])
        [6] topLight (int)
        [7] bottomLight (int)
        [8] exposure (int)

    Args:
        reflectance (List): Metadata list for the reflectance image
        transmitted_lo (List): Metadata list for the low-exposure transmitted image
        transmitted_hi (List): Metadata list for the high-exposure transmitted image

    Returns:
        None: Prints output to console.
    """
    def format_entry(label: str, data: List) -> str:
        return (
            f"{label}:\n"
            f"  Path:          {data[0]}\n"
            f"  Top Depth:     {data[1]} m\n"
            f"  Bottom Depth:  {data[2]} m\n"
            f"  Length:        {round(data[3], 2)} m\n"
            f"  Save Path:     {data[4]}\n"
            f"  Dimensions:    {data[5]}\n"
            f"  Top Light:     {data[6]}%\n"
            f"  Bottom Light:  {data[7]}%\n"
            f"  Exposure:      {data[8]}\n"
            f"  Data ID:      {data[9]}"
        )

    print(format_entry("Reflectance Image", reflectance))
    print()
    print(format_entry("Low Exposure Image", transmitted_lo))
    print()
    print(format_entry("High Exposure Image", transmitted_hi))

