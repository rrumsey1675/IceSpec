# Module for making file paths

import os

# def construct_file_paths(capture_directory, file_prefix):
#     #capture_path = os.path.join(capture_directory, "capture")
#     dir_name = os.path.basename(os.path.normpath(capture_directory))
#     hdr_path = os.path.join(capture_directory, f'{file_prefix}{dir_name}.hdr')
#     raw_path = os.path.join(capture_directory, f'{file_prefix}{dir_name}.raw')
#     output_path = os.path.join(capture_directory, f'{file_prefix}{dir_name}.nc')
#     return hdr_path, raw_path, output_path

def construct_hdr_path(capture_directory, file_prefix):
    capture_path = os.path.join(capture_directory, "capture")
    dir_name = os.path.basename(os.path.normpath(capture_directory))
    hdr_path = os.path.join(capture_path, f'{file_prefix}{dir_name}.hdr')
    return hdr_path

def construct_raw_path(capture_directory, file_prefix):
    capture_path = os.path.join(capture_directory, "capture")
    dir_name = os.path.basename(os.path.normpath(capture_directory))
    raw_path = os.path.join(capture_path, f'{file_prefix}{dir_name}.raw')
    return raw_path

def construct_nc_path(output_directory, file_prefix):
    #capture_path = os.path.join(output_directory, "capture")
    dir_name = os.path.basename(os.path.normpath(output_directory))
    nc_path = os.path.join(output_directory, f'{file_prefix}{dir_name}.nc')
    return nc_path