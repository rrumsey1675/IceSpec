# rumz // 6/25

import pandas as pd
import os
from typing import List, Tuple, Dict, Any

def getData(data_ids: List[str], csv_path: str, output_dir: str) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Retrieves metadata for a given ice core section, and selects the correct images for:
        - reflectance imaging
        - transmitted imaging (high and low exposure)

    Args:
        CoreID (str): Identifier for the specific ice core (e.g., 'ALHIC2302').
        SectionID (str): Section ID within the core (e.g., '30 Sec 34', '2404').
        csv_path (str): Path to the CSV file containing scan metadata.
        output_dir (str): Path to the base directory where outputs will be saved.

    Returns:
        Tuple of 3 coreData lists:
            - reflectance_data: Corresponds to the reflectance image
            - transmitted_lo_data: Corresponds to the lower-exposure transmitted image
            - transmitted_hi_data: Corresponds to the higher-exposure transmitted image

        Each coreData is a list with:
            [0] dataPath (str): Full path to the ENVI scan directory
            [1] top_depth (float)
            [2] bottom_depth (float)
            [3] core_length (float)
            [4] save_path (str)
            [5] px_dims (List[int])
            [6] topLight (int)
            [7] bottomLight (int)
            [8] exposure (int)
    """

    if len(data_ids) != 3:
        raise ValueError("Exactly 3 DataIDs must be provided in order: [reflectance, transmitted_lo, transmitted_hi].")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace(" ", "")
    data_subset = df[df["DataID"].isin(data_ids)]

    if data_subset.shape[0] != 3:
        raise ValueError(f"Could not find all 3 DataIDs. Found {data_subset.shape[0]} in the CSV.")

    # Keep order consistent with data_ids
    data_subset = data_subset.set_index("DataID").loc[data_ids]

    core_data_list = []

    for _, row in data_subset.iterrows():
        core_metadata = {
            "dataName": row['DirName'],
            "dataDir": row['DataDir'],
            "coreName": row['CoreID'],
            "section": row['Tube/Section'],
            "topLight": 75 if row['TopLight'] == "Dot 4" else int(row['TopLight']),
            "bottomLight": int(row['BottomLight']),
            "exposure": int(row['Exposure']),
            "top_depth": row['TopDepth'],
            "bottom_depth": row['BottomDepth'],
            "px_dims": row['PxDimensions'],
            "DataID": row.name,  # since we used set_index
            "CSV_File": os.path.basename(csv_path)
        }

        core_metadata = handleNaN(core_metadata)

        try:
            core_metadata["core_length"] = float(core_metadata["bottom_depth"]) - float(core_metadata["top_depth"])
        except ValueError:
            core_metadata["core_length"] = "Unknown"

        clean_section = core_metadata["section"].split()[0]
        core_metadata["coreName"] = core_metadata["coreName"].replace(" ", "")
        save_path = get_save_path(core_metadata["coreName"], clean_section, output_dir)

        core_metadata["px_dims"] = [int(val.strip()) for val in core_metadata["px_dims"].split(",")]

        dataPath = constructPath(core_metadata, csv_path)

        coreData = [
            dataPath,                                                           # 0
            core_metadata["top_depth"],                                         # 1
            core_metadata["bottom_depth"],                                      # 2
            core_metadata["core_length"],                                       # 3
            save_path,                                                          # 4
            core_metadata["px_dims"],                                           # 5
            core_metadata["topLight"],                                          # 6
            core_metadata["bottomLight"],                                       # 7
            core_metadata["exposure"],                                          # 8
            core_metadata["DataID"],                                            # 9
            core_metadata["coreName"],                                          # 10
            core_metadata["section"],                                           # 11
            core_metadata["CSV_File"]                                           # 12
        ]

        core_data_list.append(coreData)

    reflectance, transmitted_lo, transmitted_hi = core_data_list
    return reflectance, transmitted_lo, transmitted_hi

def handleNaN(core_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replaces missing (NaN or blank) metadata values with "Unknown" for user readability.

    Args:
        core_metadata (dict): Metadata dictionary for a core section.

    Returns:
        dict: Updated metadata with missing fields marked as "Unknown".
    """
    for key, value in core_metadata.items():
        if pd.isna(value) or value == "NaN" or value == "":
            core_metadata[key] = "Unknown"
    return core_metadata

def constructPath(core_metadata: Dict[str, Any], csv_path: str) -> None:
    """
    Constructs the full path to the data directory based on the metadata and CSV location.

    Args:
        core_metadata (dict): Metadata dictionary including dataDir and dataName.
        csv_path (str): Path to the CSV file for relative reference.

    Returns:
        dataPath (str): Full file path from root to hyperspectral image file
    """
    dataPath = os.path.join(core_metadata['dataDir'], core_metadata['dataName'])
    dataPath = os.path.join(os.path.dirname(csv_path), dataPath)

    return dataPath

def get_region_and_project(coreName: str) -> Tuple[str, str]:
    """
    Determines the region and project associated with a given ice core name.

    Args:
        coreName (str): Name of the ice core (e.g., 'ALHIC2302').

    Returns:
        tuple: (region, project) such as ('AN', 'ALHIC') or ('UNKNOWN', 'UNKNOWN').
    """

    # Mapping of core prefixes to regions
    region_map = {
        "AN": ["ALHIC", "CMC", "WDC", "SPC"],  # Antarctica
        "AL": ["DEN"],                         # Alpine
        "GL": ["GISP2", "NGRIP"]                # Greenland
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
    Constructs and ensures a save path for storing processed data.

    Args:
        coreName (str): Name of the ice core (e.g., 'ALHIC2302').
        section (str): Section identifier (e.g., '30').
        output_dir (str): Base directory for outputs.

    Returns:
        str: The full path where processed outputs should be saved.
    """

    # Get region and project from coreName
    region, project = get_region_and_project(coreName)

    # Construct save path: region/project/coreName/coreName-section
    save_path = os.path.join(output_dir, region, project, coreName, f"{coreName}-{section}")

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    return save_path

def select_low_high(exposures: List[int]) -> Tuple[int, int]:
    """
    Selects the best low and high exposure values from a list, based on silly human logic:
    - Low: the last exposure in the lower half of the distribution
    - High: the last exposure in the upper half

    Args:
        exposures (List[int]): List of exposure values.

    Returns:
        Tuple[int, int]: (low_val, high_val) based on distribution split.
    """    
    sorted_exps = sorted(exposures)
    mid = len(sorted_exps) // 2
    median = (sorted_exps[mid - 1] + sorted_exps[mid]) / 2 if len(sorted_exps) % 2 == 0 else sorted_exps[mid]

    low_group = set(e for e in exposures if e <= median)
    high_group = set(e for e in exposures if e > median)

    low_val = next(e for e in reversed(exposures) if e in low_group)
    high_val = next(e for e in reversed(exposures) if e in high_group)

    return low_val, high_val
