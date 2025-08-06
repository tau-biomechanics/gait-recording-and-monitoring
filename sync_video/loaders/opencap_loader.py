"""
OpenCap data loader module
"""

import pandas as pd
import opensim as osim


def load_opencap_data(file_path):
    """
    Load OpenCap joint angle data from .mot file
    
    Parameters:
    -----------
    file_path : str
        Path to the OpenCap .mot file
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing OpenCap data, or None if loading failed
    """
    print(f"Loading OpenCap data from: {file_path}")

    try:
        # Use OpenSim's Storage class to read the .mot file
        storage = osim.Storage(file_path)

        # Get column labels
        labels = storage.getColumnLabels()

        # Convert column labels to Python list
        column_names = []
        for i in range(labels.getSize()):
            column_names.append(labels.getitem(i))

        # Extract time points
        time_points = []
        for i in range(storage.getSize()):
            time_points.append(storage.getStateVector(i).getTime())

        # Create a dictionary to store the data
        data_dict = {}

        # Extract data for each column
        for col_idx in range(len(column_names)):
            col_name = column_names[col_idx]
            col_data = []
            for i in range(storage.getSize()):
                state_vec = storage.getStateVector(i)
                if col_idx == 0:  # time column
                    col_data.append(state_vec.getTime())
                else:
                    col_data.append(state_vec.getData().getitem(col_idx - 1))
            data_dict[col_name] = col_data

        # Convert to DataFrame
        data = pd.DataFrame(data_dict)

        print(f"Loaded OpenCap data with {len(data)} rows and {len(data.columns)} columns")
        return data
    except Exception as e:
        print(f"Error loading OpenCap data: {str(e)}")
        return None