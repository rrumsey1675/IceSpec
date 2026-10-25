# rumz

import pandas as pd

def compute_bubble_statistics(df, px_per_cm):
    """
    Compute statistical summaries for the analyzed bubble properties.

    Args:
    - df (pandas DataFrame): DataFrame containing bubble properties.
    - px_per_cm (float): Number of pixels in a cm

    Returns:
    - stats_df (pandas DataFrame): DataFrame containing mean, median, and standard deviation 
      for area, elongation, orientation, and eccentricity.
    """
    if df.empty:
        return "No valid bubbles detected."

    mean_area = round(df["area"].mean() / px_per_cm, 2)
    median_area = round(df["area"].median() / px_per_cm, 2)
    std_area = round(df["area"].std() / px_per_cm, 2)
    
    mean_elongation = round(df["elongation"].mean() / px_per_cm, 2)
    median_elongation = round(df["elongation"].median() / px_per_cm, 2)
    std_elongation = round(df["elongation"].std() / px_per_cm, 2)
    
    mean_orientation = round(df["orientation"].mean() / px_per_cm, 2)
    median_orientation = round(df["orientation"].median() / px_per_cm, 2)
    std_orientation = round(df["orientation"].std() / px_per_cm, 2)
    
    mean_eccentricity = round(df["eccentricity"].mean() / px_per_cm, 2)
    median_eccentricity = round(df["eccentricity"].median() / px_per_cm, 2)
    std_eccentricity = round(df["eccentricity"].std() / px_per_cm, 2)


    stats = {
        "Metric": ["Mean", "Median", "Std Dev"],
        "Area": [f"{mean_area} cm", f"{median_area} cm", f"{std_area} cm"],
        "Elongation": [f"{mean_elongation} %", f"{median_elongation} %", f"{std_elongation} %"],
        "Orientation": [f"{mean_orientation} degrees", f"{median_orientation} degrees", f"{std_orientation} degrees"],
        "Eccentricity": [f"{mean_eccentricity}", f"{median_eccentricity}", f"{std_eccentricity}"]
    }

    stats_df = pd.DataFrame(stats).round(2)  # Round values to 2 decimal places

    return stats_df