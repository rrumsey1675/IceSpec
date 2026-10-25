# MTF: Modular Transfer Function
This file contains python and matlab code that is used to determine the Modular Transfer Function, or Spatial Frequency Response, of our Hyperspectral Imaging System.

## How it Works
First, a printed lens test chart is imaged by the system being tested. Chart download can be found here: https://www.normankoren.com/Tutorials/MTF5.html.

Then the captured image should be uploaded to ImageJ. By using analysis tools in this software the spatial frequency can be plotted as a sine wave. This plot can be exported as a .txt file and calculated by the given SFR code.

This code is based on the work of Norman Koren: see https://www.normankoren.com/Tutorials/MTF.html