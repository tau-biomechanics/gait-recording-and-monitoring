"""
QTM data loader module
"""

import pandas as pd


def load_qtm_data(file_path):
    """
    Load QTM data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the QTM CSV file
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing QTM data, or None if loading failed
    """
    print(f"Loading QTM data from: {file_path}")
    try:
        qtm_data = pd.read_csv(file_path)
        print(f"Loaded QTM data with {len(qtm_data)} rows and {len(qtm_data.columns)} columns")

        # Convert timestamp to seconds
        if "Timestamp" in qtm_data.columns:
            qtm_data["TimeSeconds"] = qtm_data["Timestamp"] / 1000000

        return qtm_data
    except Exception as e:
        print(f"Error loading QTM data: {str(e)}")
        return None