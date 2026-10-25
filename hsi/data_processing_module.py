# data_processing_module.py
#rename this at some point

import numpy as np
import netCDF4 as nc

from hsi import data_cropping_module as datacrop
from hsi import deblending_module as deblender
from hsi import flat_field_module as flatfield
from hsi import mask_processing_module as maskprocessor
from hsi import resizing_module as resizer


def process_data(capture_directory):
    """
    Main running code. Processes hyperspectral image data through several stages:
        - Reading raw data along with corresponding dark and white referencs.
        - Crops hyperspectral data to remove edges.
        - Applies flat field correction on cropped data for each spectral band
        - Deblends the spectral data.
        - Inpaints over masked areas in each band. 
        - Conducts Principal Component Analysis (PCA) of the data.
      
    Args:
    - capture_directory (pathlib.path): The directory path where the netCDF files are located.

    Returns:
        - None.
    """
    
    ice_core_nc = capture_directory /'capture'/ f'{capture_directory.name}.nc'
    dark_ref_nc = capture_directory /'capture'/ f'DARKREF_{capture_directory.name}.nc'
    white_ref_nc = capture_directory /'capture'/ f'WHITEREF_{capture_directory.name}.nc'
    
    # Load and process data from netCDF files
    with nc.Dataset(ice_core_nc, 'r') as ds_ice_core, nc.Dataset(dark_ref_nc, 'r') as ds_dark_ref, nc.Dataset(white_ref_nc, 'r') as ds_white_ref:
        ds_ice_core.set_auto_mask(False)
        ds_dark_ref.set_auto_mask(False)
        ds_white_ref.set_auto_mask(False)
        
        ice_core_data = ds_ice_core.variables['data'][:]
        dark_ref_data = ds_dark_ref.variables['data'][:]
        white_ref_data = ds_white_ref.variables['data'][:]

        # Get user input for the removal percentage 
        #removal_percent = float(input("Enter the percentage to crop from the sides (as a decimal): ")) #This is the only user input code but for now its defaulting to 0.0
        removal_percent=0.0
        # Crop data
        representative_band_cropped, data_cropped = datacrop.data_cropping(removal_percent, ice_core_data)

        # Apply flat field correction on cropped data
        corrected_image = np.empty_like(data_cropped)
        for band in range(data_cropped.shape[2]):
            raw_image_band = data_cropped[:, :, band]
            dark_ref_band = dark_ref_data[:, :, band]
            white_ref_band = white_ref_data[:, :, band]

            # Correct the order of the dimensions for the resize_image functio
            dark_ref_resized = resizer.resize_image(dark_ref_band, (raw_image_band.shape[1], raw_image_band.shape[0]))
            white_ref_resized = resizer.resize_image(white_ref_band, (raw_image_band.shape[1], raw_image_band.shape[0]))

            # Perform flat field correction
            corrected_band =  flatfield.flat_field_correction(raw_image_band, dark_ref_resized, white_ref_resized)   
            corrected_image[:, :, band] = corrected_band

        # Deblend and inpaint each band
        individual_segm_masks = []
        fwhm_array = ds_ice_core.variables['fwhm'][:]  
        for band_index in range(corrected_image.shape[2]):
            band_data = corrected_image[:, :, band_index]
            fwhm = fwhm_array[band_index]  # Assuming fwhm_array contains FWHM values for all bands
            segm_mask = deblender.deblending_image(band_data, band_index, fwhm)
            individual_segm_masks.append(segm_mask)

        # Process data with inpainting and PCA
        inpaint_radius = 3  
        pca_transformed_data = maskprocessor.process_data_with_mask(corrected_image, individual_segm_masks, inpaint_radius)
