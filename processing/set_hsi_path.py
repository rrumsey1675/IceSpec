"""This module contains tools that are useful for setting things up path for the 
hsi processing  Jupyter notebooks.
"""

import sys
import os


def add_hsi_code_dir() -> str:
    """This function should be called at the top of Jupyter notebooks located
    in the examples folder.

    Returns:
        str: The directory that was added (hopefully the hsi-tools/code directory)
    """
    hsi_code_dir = os.path.dirname(os.getcwd())
    sys.path.insert(0, hsi_code_dir) #add a parent directory to the Python path 
    return hsi_code_dir
