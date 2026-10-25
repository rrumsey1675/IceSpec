#By Roisin Rumsey
#April 2024
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def sfrcalc(file_path, minlp=2, maxlp=200, picht=24, descr="MTF"):
    # Read data pairs from the text file
    data = np.loadtxt(file_path)

    # Extracting distance and gray values
    distance = data[:, 0]
    gray_values = data[:, 1]

    # Plot the original sine wave
    plt.figure()
    plt.plot(distance, gray_values)
    plt.xlabel('Distance (mm)')
    plt.ylabel('Gray Value')
    plt.title('Original Sine Wave: ' + descr)
    plt.grid(True)
    plt.show()

    # Find peaks
    peaks, _ = find_peaks(gray_values, prominence=10)  # Adjust prominence as needed
    valleys, _ = find_peaks(-gray_values, prominence=10) #Adjust prominence as needed

    plt.figure()
    plt.plot(peaks, gray_values[peaks], "xr"); plt.plot(gray_values)
    plt.plot(valleys, gray_values[valleys], "vg"); plt.legend(['Peak','Gray Value','Valleys'])
    plt.title('Original Sine Wave: Peaks')
    plt.grid(True)
    plt.show()

    # calculate C0
    Vw = gray_values[peaks[0]]
    Vb = gray_values[valleys[1]]
    C0 = (Vw-Vb) / (Vw+Vb)

    # set up Cf array
    Cf = np.zeros(len(peaks))
    k=0

    # calculate C(f)
    for i,j in zip(peaks,valleys):
        Cf[k] = (gray_values[i]-gray_values[j]) / (gray_values[i]+gray_values[j])
        k+=1
        
    #### plot spatial frequency response (aka modulation transfer function)
    xstar = np.linspace(0,1,num=k+1)   # normalized chart distance scale
    #plt.plot(xstar,Cf)

    # set up spatial frequency scale
    sf1 = minlp   # lp/mm
    sf2 = maxlp # lp/mm
    sf = sf1 * (sf2/sf1)**xstar
    sf = sf[:len(Cf)]

    print("sf: ", len(sf))
    print("Cf: ", len(Cf))

    plt.figure()
    plt.plot(sf,Cf)
    plt.xlabel('Line pairs per millimeter (lp/mm)')
    plt.ylabel('Spatial frequency response')
    plt.title('System Description: ' + descr)
    plt.grid(True)
    plt.show()