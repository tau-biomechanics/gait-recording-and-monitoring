"""
Force data processing functions
"""

import numpy as np
import pandas as pd


def extract_qtm_force_data(qtm_data):
    """
    Extract force data from QTM data
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing force data
    """
    # Try to find force columns in QTM data
    force_cols = [
        col for col in qtm_data.columns if "force" in col.lower() and col.endswith("Z")
    ]

    if not force_cols:
        print("No force Z columns found in QTM data, looking for any force columns")
        force_cols = [col for col in qtm_data.columns if "force" in col.lower()]

    if not force_cols:
        print("No force columns found in QTM data")
        # Return placeholder data
        return pd.DataFrame(
            {"TimeSeconds": qtm_data["TimeSeconds"], "Force": np.zeros(len(qtm_data))}
        )

    # Check if we can calculate resultant force from XYZ components
    if all(col in qtm_data.columns for col in ["ForcePlate1_ForceX", "ForcePlate1_ForceY", "ForcePlate1_ForceZ"]):
        print("Calculating resultant force from X, Y, Z components")
        return _calculate_resultant_force(qtm_data)
    else:
        # Choose the force column to use (may need adjustment based on data)
        force_col = force_cols[0]
        print(f"Using QTM force column: {force_col}")
        
        # Return time and the selected force column
        return pd.DataFrame(
            {
                "TimeSeconds": qtm_data["TimeSeconds"],
                "Force": pd.to_numeric(qtm_data[force_col], errors="coerce")
                .fillna(0)
                .values,
            }
        )


def _calculate_resultant_force(qtm_data):
    """
    Calculate resultant force from X, Y, Z components
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM data with force components
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing resultant force
    """
    forceX = (
        pd.to_numeric(qtm_data["ForcePlate1_ForceX"], errors="coerce")
        .fillna(0)
        .values
    )
    forceY = (
        pd.to_numeric(qtm_data["ForcePlate1_ForceY"], errors="coerce")
        .fillna(0)
        .values
    )
    forceZ = (
        pd.to_numeric(qtm_data["ForcePlate1_ForceZ"], errors="coerce")
        .fillna(0)
        .values
    )

    # Calculate the resultant force
    resultant_force = np.sqrt(forceX**2 + forceY**2 + forceZ**2)

    # Create dataframe with time and resultant force
    force_data = pd.DataFrame(
        {"TimeSeconds": qtm_data["TimeSeconds"], "Force": resultant_force}
    )

    return force_data