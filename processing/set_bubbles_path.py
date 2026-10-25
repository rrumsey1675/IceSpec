import sys
import os


def add_bubble_code_dir() -> str:
    """This function should be called at the top of Jupyter notebooks located
    in the examples folder.

    Returns:
        str: The directory that was added (hopefully the stat/bubbles directory)
    """
    bubble_code_dir = os.path.dirname(os.getcwd())
    sys.path.insert(0, bubble_code_dir) #add a parent directory to the Python path 
    return bubble_code_dir