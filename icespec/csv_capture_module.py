# rumz // 3/14/25

import pandas as pd
import os

def getData(DataID: str, csv_path: str, output_dir: str) -> str:
    """
    Retrieves the file path of the ENVI dataset corresponding to a given ice core sample.

    Args:
        DataID (str): Identifier for the specific ice core sample.
        csv_path (str): Path to the csv file in directory containing metadata.
        output_dir (str): Path to base output directory.

    Returns:
        fullname(str), core_metadata["top_depth"](str), core_metadata["bottom_depth"](str), core_metadata["core_length"](str), save_path(str), 
        core_metadata["px_dims"](str), core_metadata(dict)
    """

    csv_path = csv_path #set path to processing folder
    DataFrame = pd.read_csv(csv_path) #reading in file containing core data metadata

    
    data = DataFrame[(DataFrame['DataID'] == DataID)] # grabs row for chosen image
    data.columns = data.columns.str.replace(" ", "")
    
    core_metadata = {
        "dataName": data.iloc[0]['DirName'],            # Folder name where the scan is stored
        "dataDir": data.iloc[0]['DataDir'],             # Base directory containing the core folder
        "coreName": data.iloc[0]['CoreID'],             # Unique identifier for the ice core sample
        "section": data.iloc[0]['Tube/Section'],        # Section number of the core (e.g., "38")
        "topLight": data.iloc[0]['TopLight' or 'Top Light'],           # Top light intensity setting (e.g., "Dot 4" or percentage)
        "bottomLight": data.iloc[0]['BottomLight'],     # Bottom light intensity (usually a percentage)
        "exposure": data.iloc[0]['Exposure'],           # Exposure value used during image capture
        "top_depth": data.iloc[0]['TopDepth'],          # Depth at the top of this core section (in meters)
        "bottom_depth": data.iloc[0]['BottomDepth'],    # Depth at the bottom of this core section (in meters)
        "px_dims": data.iloc[0]['PxDimensions'],         # Image dimensions (e.g., "1290, 6750, 100, 600")
        "TopWR100": data.iloc[0]['TopWR100'] if 'TopWR100' in data.columns else None,
        "BottomWR100": data.iloc[0]['BottomWR100'] if 'BottomWR100' in data.columns else None,
        "CoreBottom": data.iloc[0]['CoreBottom'] if 'CoreBottom' in data.columns else None,
        "DataID": DataID,
        "CSV_File": os.path.basename(csv_path)
    }

    # Check for missing file path information
    if any(pd.isna(core_metadata[key]) or str(core_metadata[key]).strip() == "NaN" or core_metadata[key] == ""
           for key in ["dataName", "dataDir"]):
        raise ValueError("Error: Missing file path information in CSV.")

    core_metadata = handleNaN(core_metadata)

    # set top light setting to int for percentage / increase user readability
    if core_metadata["topLight"] == "Dot 4":
        core_metadata["topLight"] = 50

    # calculate length from depths
    try:
        core_metadata["core_length"] = float(core_metadata["bottom_depth"]) - float(core_metadata["top_depth"])
    except ValueError:
        core_metadata["core_length"] = "Unknown"

    # print information about chosen core to user
    print("Core: ", core_metadata["coreName"])
    print("Section: ", core_metadata["section"])
    print(f"Lighting: Top light - {core_metadata["topLight"]}%, Bottom Light - {int(core_metadata["bottomLight"])}%")
    print("Exposure: ", int(core_metadata["exposure"]))
    print("Top Depth: ", core_metadata["top_depth"])
    print("Bottom Depth: ", core_metadata["bottom_depth"])
    print("Length = ", core_metadata["core_length"])
    print("Dimensions: ", core_metadata["px_dims"])

    # Generate full directory path for traversal to hsi scan
    fullname = f"{core_metadata['dataDir']}/{core_metadata['dataName']}"
    print("Data Path: ", fullname)

    
    ## For Save Path    
    # Clean section and coreName for save path
    if os.path.basename(csv_path) == "HSI_Sept2025.csv":
        clean_section = core_metadata["section"].split()[1]
    else:
        clean_section = core_metadata["section"].split()[0]
    core_metadata["coreName"] = core_metadata["coreName"].replace(" ", "")

    if clean_section == "51_2":
        clean_section = "51"

    # Construct save path format: 'region/project/coreName/coreName-section'
    save_path = get_save_path(core_metadata["coreName"], clean_section, output_dir)


    return fullname, core_metadata["top_depth"], core_metadata["bottom_depth"], core_metadata["core_length"], save_path, core_metadata["px_dims"], core_metadata

def handleNaN(core_metadata):
    """
    Checks for missing (NaN) values in the provided parameters and prompts the user to continue or abort.

    Args:
        core_metadata (dictionary): Contains metadata on the specfic ice core

    Returns:
        core_metadata (dictionary): Updated to show "Unknown" for empty data
    """
    for key, value in core_metadata.items():
        if pd.isna(value) or value == "NaN" or value == "":
            core_metadata[key] = "Unknown"
    return core_metadata

def get_region_and_project(coreName: str) -> tuple:
    """
    Determines the region and project based on the coreName prefix.

    Parameters:
    - coreName (str): The core name (e.g., 'ALHIC2302', 'GISP2').

    Returns:
    - tuple: (region, project) - e.g., ('AN', 'ALHIC')
    """

    # Mapping of core prefixes to regions
    region_map = {
        "AN": ["ALHIC", "CMC", "WDC", "SPC"],  # Antarctica
        "AL": ["DEN", "DIC1"],                 # Alpine
        "GL": ["GISP2", "NGRIP"]               # Greenland
    }

    # Special case: Handle ALHIC Firn core
    if coreName == "Firn":
        return "AN", "ALHIC"
    
    # Identify project and region by matching the prefix
    for region, projects in region_map.items():
        for project in projects:
            if coreName.startswith(project):
                return region, project
    
    # Default case if no match is found
    return "UNKNOWN", "UNKNOWN"

def get_save_path(coreName: str, section: str, output_dir: str) -> str:
    """
    Constructs the save path using region, project, coreName, and section.

    Parameters:
    - coreName (str): Name of the ice core (e.g., 'ALHIC2302').
    - section (str): Core section (e.g., '30').
    - output_dir (str): Base output directory.

    Returns:
    - str: Full save path
    """

    # Get region and project from coreName
    region, project = get_region_and_project(coreName)

    # Construct save path: region/project/coreName/coreName-section
    save_path = os.path.join(output_dir, region, project, coreName, f"{coreName}-{section}")

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    return save_path
