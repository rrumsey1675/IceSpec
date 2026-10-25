#rumz // 4/25

import os
import matplotlib.pyplot as plt
import spectral
spectral.settings.envi_support_nonlowercase_params = True
import spectral.io.envi as envi
import numpy as np

def capture(data_path: str, Debug: bool):
    """
    Loads hyperspectral ENVI data along with white and dark reference images.

    Args:
        data (str): Path to the ENVI dataset.
        Debug (bool, optional): Enables additional previewing if True.

    Returns:
        tuple: (data_nparr (np.ndarray), w_arr (np.ndarray), d_arr (np.ndarray), fwhm (np.ndarray))
    """

    os.environ['SPECTRAL_DATA'] = data_path + '/capture'
    if Debug == True:
        print("Spectral Environ: ", os.environ['SPECTRAL_DATA'])

    dsetName = os.path.basename(data_path)
    fullPath = data_path + '/capture/' + dsetName
    data_captured = envi.open(dsetName + '.hdr')
    fwhm = data_captured.bands.bandwidths
    w_ref=envi.open('WHITEREF_'+ dsetName + '.hdr')
    d_ref=envi.open('DARKREF_'+ dsetName + '.hdr')
    meta = data_captured.metadata
    wvls = meta.get('Wavelength')
    wvls = [float(i) for i in wvls]
    wvls = np.array(wvls)
    print(type(wvls))
    data_nparr = np.array(data_captured.load())
    w_arr = np.array(w_ref.load())
    d_arr = np.array(d_ref.load())
    if Debug == True:
        print(f"Data: {data_nparr.shape}, W ref: {w_arr.shape}, D ref: {d_arr.shape}")

    return data_nparr, w_arr, d_arr, fwhm, wvls