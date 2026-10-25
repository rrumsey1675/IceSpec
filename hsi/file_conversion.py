# 2/27/24 - rumz
#created to quickly convert given images to our usable netcdf files within process code
#pretty much just a copy/paste of Turnkey_conversion in examples

import sys
#sys.path.append('../')
from hsi import makeFilePath as mf
import numpy as np
import re
from spectral import envi
from netCDF4 import Dataset as nc
from pathlib import Path
from pathlib import PurePath


# Function to read .hdr metadata and return it along with interleave type
def read_hdr_metadata(hdr_path):
   metadata = {}
   fwhm_values=[]
   inside_fwhm_block=False
  
   with open(hdr_path, 'r') as file:
       for line in file:
           line=line.strip()
           if line.strip().lower().startswith("fwhm = {"):
               inside_fwhm_block= True
           elif inside_fwhm_block and line.strip().endswith('}'):
               inside_fwhm_block= False
               fwhm_values.extend([float(value.strip()) for value in line.rstrip('}').split(',') if value])
               break
           elif inside_fwhm_block:
               fwhm_values.extend([float(value.strip()) for value in line.split(',') if value])
               continue


           match = re.match(r'(\w+)\s*=\s*(.+)', line)
           if match:
               key, value = match.groups()
               key = key.strip()  # Keep the original key including case and spaces
               value = value.strip()


               if value.startswith('{') and value.endswith('}'):
                   value = value[1:-1].split(',')  # Process value lists enclosed in curly braces
               metadata[key] = value


   # Convert the interleave and data type to lower-case for consistent handling later on
   interleave = metadata.get("interleave", "bil").lower()
   data_type = int(metadata.get("data type",12))


   return metadata, interleave, data_type, fwhm_values
def envi_data_type_to_numpy(envi_data_type):
   data_type_map = {
       1: np.uint8,   # Byte
       2: np.int16,   # Integer
       3: np.int32,   # Long
       4: np.float32, # Floating-point
       5: np.float64, # Double-precision
       6: np.complex64, # Complex
       9: np.complex128, # Double-precision complex
       12: np.uint16, # Unsigned integer
       13: np.uint32, # Unsigned long integer
       14: np.int64,  # 64-bit long integer (signed)
       15: np.uint64, # 64-bit unsigned long integer (unsigned)
   }
   return data_type_map.get(envi_data_type, None)

# Function to convert ENVI data to NetCDF format
def convert_envi_to_netcdf(hdr_path, raw_path, nc_path):
   metadata, interleave, envi_data_type, fwhm_values = read_hdr_metadata(hdr_path)
   lines = int(metadata['lines'])
   samples = int(metadata['samples'])
   bands = int(metadata['bands'])
   data_type = envi_data_type_to_numpy(envi_data_type)
  
   if data_type is None:
       raise ValueError(f"Unsupported ENVI data type: {envi_data_type}")


   with open(raw_path, 'rb') as file:
       data = np.fromfile(file, dtype=data_type)
       if interleave == "bsq":
           data = data.reshape((bands, lines, samples))
           data = np.transpose(data, (1, 2, 0))
       elif interleave == "bil":
           data = data.reshape((lines, bands, samples))
           data = np.transpose(data, (0, 2, 1))
       elif interleave == "bip":
           data = data.reshape((lines, samples, bands))
       else:
           raise ValueError(f"Unsupported interleaving method: {interleave}")


   with nc(nc_path, 'w') as dataset:
       dataset.createDimension('lines', lines)
       dataset.createDimension('samples', samples)
       dataset.createDimension('bands', bands)


       if fwhm_values:
           dataset.createDimension('fwhm_dim', len(fwhm_values))
           fwhm_var =dataset.createVariable("fwhm","f4",("fwhm_dim",))
           fwhm_var[:] =fwhm_values


       var = dataset.createVariable('data', data_type, ('lines', 'samples', 'bands'))
       var[:] = data


       # Set metadata attributes
       for key, value in metadata.items():
           setattr(dataset, key, value)


   #print(f'NetCDF file created at: {nc_path}')


# Function to process the main data
def process_data(capture_directory, file_prefix, output_directory):
   #hdr_path, raw_path, output_path = construct_file_paths(capture_directory, file_prefix)
   hdr_path = mf.construct_hdr_path(capture_directory,file_prefix)
   raw_path = mf.construct_raw_path(capture_directory,file_prefix)
   nc_path = mf.construct_nc_path(output_directory,file_prefix)
   convert_envi_to_netcdf(hdr_path, raw_path, nc_path)
   
   return nc_path