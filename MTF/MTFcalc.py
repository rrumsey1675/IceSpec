import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sfrcalc(inpfile, minlp=2, maxlp=200, picht=24):
    """
    Calculate SFR or MTF from an Excel-formatted text data file from ImageJ.
    
    Parameters:
    inpfile : str : Path to the input file.
    minlp : float : Minimum frequency (lp/mm).
    maxlp : float : Maximum frequency (lp/mm).
    picht : float : Picture height (mm).
    """

    descr = input('Enter system description: ')
    
    # Read the input data file
    try:
        arrinp = pd.read_csv(inpfile, header=None, delim_whitespace=True).values.flatten()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"{len(arrinp)} data points read in")
    
    # Data in the first column is unreliable-- limited precision. Throw it out.
    arr2 = arrinp[1::2]
    l_ar = len(arr2)
    
    print(f"{len(arr2)} data points after filtering")
    
    # Detect peaks
    dif1 = np.diff(arr2)
    d1 = (dif1 > 0)  # 1 or 0
    d1 = (d1[1:] != d1[:-1])  # 1 at peaks; 0 elsewhere
    peak_loc = np.where(d1)[0] + 1  # integer peak locations
    n_peaks = len(peak_loc)  # number of detected peaks
    
    print(f"{n_peaks} detected peaks")
    peak_ampl = arr2[peak_loc]  # End peak detection

    if n_peaks < 2:
        print("Not enough peaks detected. End program.")
        return

    # Calculate the response
    summ = peak_ampl[:-1] + peak_ampl[1:]
    resp1 = np.abs(np.diff(peak_ampl)) / summ
    resp1 = resp1 / np.max(resp1)  # Normalize

    # Frequency scale
    lpratio = maxlp / minlp
    xax1 = np.arange(l_ar) / (l_ar - 1)
    lpmm1 = minlp * lpratio ** xax1
    
    # Plotting
    plt.figure(figsize=(8, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(lpmm1, arr2)
    plt.grid(True)
    plt.title(descr)
    plt.xlabel('Line pairs per millimeter (lp/mm)')
    plt.ylabel('Input data')
    
    xax2 = peak_loc[1:] / (l_ar - 1)  # X-axis for MTF
    lpmm2 = minlp * lpratio ** xax2

    plt.subplot(3, 1, 2)
    plt.plot(lpmm2, resp1)
    plt.grid(True)
    plt.xlabel('Line pairs per millimeter (lp/mm)')
    plt.ylabel('Spatial frequency response')

    if picht > 0:
        plt.subplot(3, 1, 3)
        plt.plot(lpmm2 * 2 * picht, resp1)
        plt.grid(True)
        plt.xlabel(f'Line widths per picture height (lw/ph) for ph = {picht} mm')
        plt.ylabel('Spatial frequency response')
    
    plt.tight_layout()
    plt.show()
