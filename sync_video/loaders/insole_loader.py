"""
Insole data loader module
"""

import pandas as pd
import numpy as np
import csv


def load_insole_data(file_path):
    """
    Load insole data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the insole CSV file
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing insole data, or None if loading failed
    """
    print(f"Loading insole data from: {file_path}")
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read headers

            # Extract only the data we need
            timestamps = []
            left_forces = []  # Will store data from column Z (index 25)
            right_forces = [] # Will store data from column AY (index 50)
            
            # Define target column indices
            timestamp_col_idx = 0
            left_force_col_idx = 25  # Excel column Z
            right_force_col_idx = 50  # Excel column AY

            for row_idx, row in enumerate(reader):
                if not row:  # Skip empty rows
                    continue

                # Check if row has enough columns
                if len(row) > max(timestamp_col_idx, left_force_col_idx, right_force_col_idx):
                    try:
                        timestamps.append(float(row[timestamp_col_idx]))
                        left_forces.append(float(row[left_force_col_idx]))
                        right_forces.append(float(row[right_force_col_idx]))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse row {row_idx+1} for required columns. Error: {e}. Row snippet: {row[:5]}...")
                        # Append NaN or handle appropriately if needed
                        timestamps.append(np.nan)
                        left_forces.append(np.nan)
                        right_forces.append(np.nan)
                        continue
                else:
                    print(f"Warning: Row {row_idx+1} has only {len(row)} columns, skipping.")

            # Create a DataFrame with just the columns we need
            insole_data = pd.DataFrame(
                {
                    "Timestamp": timestamps,
                    "Left_Force": left_forces,
                    "Right_Force": right_forces,
                }
            )
            
            # Remove rows where parsing failed (NaNs)
            insole_data.dropna(inplace=True)
            
            if insole_data.empty:
                print("Error: No valid insole data rows found after parsing.")
                return None

            # Convert timestamp to relative time in seconds
            t0 = insole_data["Timestamp"].iloc[0]
            insole_data["TimeSeconds"] = insole_data["Timestamp"] - t0

            print(f"Loaded insole data with {len(insole_data)} valid rows")
            return insole_data
    except Exception as e:
        print(f"Error loading insole data: {str(e)}")
        return None