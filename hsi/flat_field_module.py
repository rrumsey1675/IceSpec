"""This needs more work.
The apparent structure and variation in signal intensity of these spectra is due to the wavelength dependencies
of both the sample illumination source and the sensitivity of the detector used.
It is generally more useful to correct or transform raw signal data into reflectance or absorbance units
by comparing with spectra of standard reference materials. 
The usual transformation to reflectance values is obtained by:
correcting sample spectra (C) (raw data)
for detector dark current (Dk) (Dark)
and dividing by a similarly corrected total reflectance spectrum WRef (Ref100), Ref50 or Ref20
Equation (1) transforms instrument measurement values in A/D counts ( C, Dk, and WRef)
(C-Dk)/(WRef-Dk) into unitless reflectance values. This transform is often termed a one-point calibration
"""

import numpy as np

def flat_field_correction(R, D, F):
    """
    Computes Flat field correction on the dataset using the white reference and dark references. 
    
    Args:
    - R (2D np.ndarray Float 32): Raw image band data, in a float 32 numpy array.
    - D (2D np.ndarray Float 32): Dark field image band data (Dark ref), in a float 32 numpy array.
    - F (2D np.ndarray Float 32): Flat field image band data (White ref), in a float 32 numpy array.
    
    Returns:
        - C (2D np.ndarray Float 32): Corrected image band data, in a float 32 numpy array.
    """
    
    epsilon = 1e-6   #Prevents division by 0
    m = np.mean(F - D) #Image-averaged value of (F-D)
    denominator = F - D #Used to simplify the later equations
    denominator = np.where(denominator < epsilon, epsilon, denominator) #Any value within the denominator array that is less than epsilon is changed to equal epsilon.
    G = m / denominator #G: Gain, allows for an easier reading of the flat field correction equation.
    C = (R - D) * G #Final equation to create the corrected image.
    return C #Corrected image.