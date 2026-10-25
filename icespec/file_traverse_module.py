# rumz 5/21/24
#
# CSV data is taken and used to traverse the given scans of a core section

def getScans(scanNum: int, corePath: str):
   """
    Retrieves file paths for multiple scans of a given ice core sample. 
    ***This is in use until Denali scans are added to HSIJuly.csv***

    Args:
        scanNum (int): The number of scans to retrieve.
        corePath (str): The directory containing the core scans.

    Returns:
        list: Paths to the selected ENVI scan files.
    """

   datasets = []
    
   for i in range((scanNum)):
      datasets.append("{}{:04d}".format(corePath, i+1))

   return datasets