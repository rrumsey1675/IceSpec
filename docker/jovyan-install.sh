#!/usr/bin/env bash
######################################################################
# This file is read-only. To edit, change the README.org file.
######################################################################

conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda install -y testbook spectral astropy photutils opencv netcdf4 \
      xarray
