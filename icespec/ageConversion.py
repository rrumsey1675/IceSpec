## rumz 3/25

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def map_depths_to_ages(timescale_path: str, cm_depths: np.ndarray, top_depth: float, depth_col: str, age_col: str, header: str = 0) -> pd.DataFrame:
    """
    Maps known cm-scale depths to estimated ages using linear interpolation
    based on a depth–year timescale loaded from an Excel file.

    Args:
        timescale_path (str): Path to the Excel file with 'depth_m' and 'year_BP' columns.
        cm_depths (np.ndarray): 1D array of known depths (in meters) to map to ages.
        top_depth: Depth in meters of the top of core section.
        depth_col: Name of the depth column in the Excel file.
        age_col: Name of the ice age column in the Excel file.
        header: Row index (0-based) where the header is located in the Excel file.

    Returns:
        pd.DataFrame: A dataframe with columns 'Depth' and 'Ice Age'.
    """

    # Load timescale data
    timescale = pd.read_excel(
        timescale_path,
        header=header,
        usecols=[depth_col, age_col]
    ).dropna(subset=[depth_col, age_col])

    # Ensure numeric types
    timescale[depth_col] = pd.to_numeric(timescale[depth_col], errors="coerce")
    timescale[age_col] = pd.to_numeric(timescale[age_col], errors="coerce")
    timescale = timescale.dropna(subset=[depth_col, age_col])

    # Sort by depth just in case
    timescale = timescale.sort_values(by=depth_col)

    depths_m = [cm / 100 for cm in cm_depths]

    # Interpolate
    interp_ages = np.round(np.interp(depths_m, timescale[depth_col], timescale[age_col]),2)

    # Create and return a DataFrame
    result_df = pd.DataFrame({
        "cm_depth": cm_depths,
        "ice_age_years": interp_ages
    })

    return result_df
    
    # # 1. Load the timescale data from Excel
    # timescale_df = pd.read_excel(timescale_path, header=40, usecols=["Depth", "Ice Age"])
    # timescale_df = timescale_df.iloc[1:]
    # print(timescale_df)
    # timescale_df["Depth"] = pd.to_numeric(timescale_df["Depth"], errors="coerce")
    # timescale_df["Ice Age"] = pd.to_numeric(timescale_df["Ice Age"], errors="coerce")


    # # 2. Extract depth and age arrays
    # depths = timescale_df["Depth"]
    # ages = timescale_df["Ice Age"]
    
    # # 3. Create an interpolation function (linear)
    # depth_to_age = interp1d(depths, ages, kind='linear', fill_value='extrapolate')
    
    # # 4. Apply interpolation to cm depths
    # cm_ages = depth_to_age(cm_depths)
    
    # # 5. Return a dataframe of depth and age
    # age_map_df = pd.DataFrame({
    #     "depth_cm": cm_depths,
    #     "age_BP": np.round(cm_ages,2)
    # })
    
    # return age_map_df
