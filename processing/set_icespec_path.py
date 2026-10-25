"""This module contains tools that are useful for setting things up path for the 
icespec processing  Jupyter notebooks.
"""

import sys
import os


def add_icespec_code_dir() -> str:
    """This function should be called at the top of Jupyter notebooks located
    in the examples folder.

    Returns:
        str: The directory that was added (hopefully the stat/icespec directory)
    """
    icespec_code_dir = os.path.dirname(os.getcwd())
    sys.path.insert(0, icespec_code_dir) #add a parent directory to the Python path 
    return icespec_code_dir