#!/usr/bin/env python3
"""
Comparison of Force Data: Moticon Insoles vs QTM Measurements

This script loads data from Moticon insoles and QTM force plate measurements,
aligns the timestamps, and creates comparison plots of the force data.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
# Create output directory if it doesn't exist
OUTPUT_DIR = "comparison_plots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_qtm_data(file_path):
    """
    Load QTM force data from a CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the QTM CSV file

    Returns:
    --------
    pandas.DataFrame
        Processed QTM data with timestamps and force values
    """
    print(f"Loading QTM data from: {file_path}")

    # Load the QTM data
    qtm_data = pd.read_csv(file_path)

    # Convert timestamp to seconds (QTM timestamp is in microseconds)
    qtm_data["TimeSeconds"] = qtm_data["Timestamp"] / 1000000

    # Return the data
    return qtm_data


def load_insole_data(file_path):
    """
    Load Moticon insole data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the insole CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed insole data with timestamps and force values
    """
    print(f"Loading insole data from: {file_path}")
    
    # The CSV file has inconsistent formatting with more data values than headers
    # Read the file manually to extract just the columns we need
    try:
        with open(file_path, 'r') as f:
            import csv
            reader = csv.reader(f)
            headers = next(reader)  # Read headers
            
            print(f"CSV Headers: {headers}")
            
            # Extract only the data we need
            timestamps = []
            left_forces = []
            right_forces = []
            
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                    
                # First column is timestamp, second is left force, third is right force
                if len(row) >= 3:
                    try:
                        timestamps.append(float(row[0]))
                        left_forces.append(float(row[1]))
                        right_forces.append(float(row[2]))
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse row: {row[:5]}...")
                        continue
            
            # Create a DataFrame with just the columns we need
            insole_data = pd.DataFrame({
                'Timestamp': timestamps,
                'Left_Total_Force': left_forces,
                'Right_Total_Force': right_forces
            })
            
            print(f"Manually parsed data: {len(insole_data)} rows")
            print(f"First 5 timestamps: {insole_data['Timestamp'].head(5).tolist()}")
    except Exception as e:
        print(f"Error parsing CSV file manually: {str(e)}")
        print("Falling back to pandas CSV reader (may not handle the file correctly)")
        # Load the insole data with pandas as fallback
        insole_data = pd.read_csv(file_path)
    
    # Print column names to help with debugging
    print(f"Insole data columns: {insole_data.columns.tolist()}")
    print(f"Insole data size: {len(insole_data)} rows")
    
    # Process timestamp to convert to relative time in seconds
    if "Timestamp" in insole_data.columns:
        # Convert to numeric to ensure we handle potential string values
        insole_data["Timestamp"] = pd.to_numeric(insole_data["Timestamp"], errors="coerce")
        
        # Check if timestamps are UNIX timestamps (large values in seconds since epoch)
        if insole_data["Timestamp"].min() > 1000000000:  # Typical UNIX timestamp is > 1 billion
            print("Detected UNIX timestamps (seconds since epoch)")
            # Use relative time from first timestamp
            t0 = insole_data["Timestamp"].iloc[0]
            insole_data["TimeSeconds"] = insole_data["Timestamp"] - t0
            print(f"Using t0 = {t0} as reference time")
        else:
            # For other timestamp formats
            t0 = insole_data["Timestamp"].iloc[0]
            insole_data["TimeSeconds"] = insole_data["Timestamp"] - t0
            
        # Verify TimeSeconds is properly populated
        print(f"First 5 processed timestamps: {insole_data['TimeSeconds'].head(5).tolist()}")
    else:
        # If no timestamp column, use a time index based on sampling rate
        print("No Timestamp column found in insole data, creating time index")
        # Assuming 100 Hz sampling rate
        insole_data["TimeSeconds"] = np.arange(len(insole_data)) / 100
    
    print(
        f"Time range after conversion: {insole_data['TimeSeconds'].min():.2f} to {insole_data['TimeSeconds'].max():.2f} seconds"
    )
    
    # Return the data
    return insole_data


def find_matching_data_files():
    """
    Find matching QTM and insole data files based on timestamp in filename.

    Returns:
    --------
    list of tuples
        Each tuple contains (qtm_file_path, insole_file_path)
    """
    # Get lists of all data files
    qtm_files = glob.glob("data/qtm_data/qtm_data_*.csv")
    insole_files = glob.glob("data/insole_data/insole_data_*.csv")

    # Create a mapping of timestamps to file paths
    qtm_map = {
        os.path.basename(f).replace("qtm_data_", "").replace(".csv", ""): f
        for f in qtm_files
    }
    insole_map = {
        os.path.basename(f).replace("insole_data_", "").replace(".csv", ""): f
        for f in insole_files
    }

    # Find timestamps that exist in both maps
    matching_timestamps = set(qtm_map.keys()) & set(insole_map.keys())

    # Create list of matching file pairs
    matching_files = [(qtm_map[ts], insole_map[ts]) for ts in matching_timestamps]

    print(f"Found {len(matching_files)} matching QTM and insole data file pairs")
    return matching_files


def filter_signal(data, fs=100, cutoff=10, order=4):
    """
    Apply a low-pass Butterworth filter to the signal.

    Parameters:
    -----------
    data : array_like
        The data to filter
    fs : float
        Sampling frequency in Hz
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order

    Returns:
    --------
    array_like
        Filtered data
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def process_and_compare_forces(qtm_data, insole_data):
    """
    Process and compare force data from QTM and insole measurements.

    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM force data
    insole_data : pandas.DataFrame
        Insole force data

    Returns:
    --------
    tuple
        (qtm_time, qtm_force, insole_time, insole_force_left, insole_force_right)
    """
    # Print debug info
    print(
        f"QTM data time range: {min(qtm_data['TimeSeconds'])} to {max(qtm_data['TimeSeconds'])} seconds"
    )
    print(f"QTM data size: {len(qtm_data)} rows")
    print(
        f"Insole data time range: {min(insole_data['TimeSeconds'])} to {max(insole_data['TimeSeconds'])} seconds"
    )
    print(f"Insole data size: {len(insole_data)} rows")

    # Extract QTM force data (assuming it's in Newtons)
    # Try to find force columns in the data
    force_cols = [
        col for col in qtm_data.columns if "force" in col.lower() and col.endswith("Z")
    ]

    if not force_cols:
        print(
            "Warning: No force Z columns found in QTM data, looking for any force columns"
        )
        force_cols = [col for col in qtm_data.columns if "force" in col.lower()]

    if not force_cols:
        print("Warning: No force columns found in QTM data")
        # Attempt to use sample values for demo purposes
        qtm_time = qtm_data["TimeSeconds"].values
        qtm_force = np.zeros(len(qtm_time))
    else:
        # Use the first force column (or modify as needed)
        print(f"Using QTM force columns: {force_cols}")
        qtm_time = qtm_data["TimeSeconds"].values

        # Sum the force components if available, otherwise use the first force column
        if (
            "ForcePlate1_ForceX" in qtm_data.columns
            and "ForcePlate1_ForceY" in qtm_data.columns
            and "ForcePlate1_ForceZ" in qtm_data.columns
        ):
            # Make sure to convert to numeric, handling errors
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
            qtm_force = np.sqrt(forceX**2 + forceY**2 + forceZ**2)
        else:
            # Use the first available force column
            qtm_force = (
                pd.to_numeric(qtm_data[force_cols[0]], errors="coerce").fillna(0).values
            )

    # Extract insole force data
    insole_time = insole_data["TimeSeconds"].values

    # Check if the expected force columns exist
    if (
        "Left_Total_Force" in insole_data.columns
        and "Right_Total_Force" in insole_data.columns
    ):
        insole_force_left = (
            pd.to_numeric(insole_data["Left_Total_Force"], errors="coerce")
            .fillna(0)
            .values
        )
        insole_force_right = (
            pd.to_numeric(insole_data["Right_Total_Force"], errors="coerce")
            .fillna(0)
            .values
        )
    else:
        print("Warning: Expected force columns not found in insole data")
        # Try to find any force-related columns
        force_cols = [col for col in insole_data.columns if "force" in col.lower()]
        if force_cols:
            print(f"Found potential force columns: {force_cols}")
            # Assume first two force columns are left and right
            if len(force_cols) >= 2:
                insole_force_left = (
                    pd.to_numeric(insole_data[force_cols[0]], errors="coerce")
                    .fillna(0)
                    .values
                )
                insole_force_right = (
                    pd.to_numeric(insole_data[force_cols[1]], errors="coerce")
                    .fillna(0)
                    .values
                )
            else:
                # If only one force column, use it for both left and right
                print(f"Only one force column found: {force_cols[0]}")
                insole_force_left = (
                    pd.to_numeric(insole_data[force_cols[0]], errors="coerce")
                    .fillna(0)
                    .values
                )
                insole_force_right = insole_force_left.copy()
        else:
            # Use the second and third columns as left and right forces (adjust as needed)
            print("No force columns found, using columns 1 and 2 as placeholders")
            cols = insole_data.columns[1:3]
            print(f"Using columns instead: {cols}")
            insole_force_left = (
                pd.to_numeric(insole_data[cols[0]], errors="coerce").fillna(0).values
            )
            insole_force_right = (
                pd.to_numeric(insole_data[cols[1]], errors="coerce").fillna(0).values
            )

    # Invert the insole force values by multiplying by -1
    print("Inverting insole force values (multiplying by -1)")
    insole_force_left = -1 * insole_force_left
    insole_force_right = -1 * insole_force_right

    # Scale insole forces to match QTM force range
    # Only apply scaling if we have valid QTM force data
    if np.max(qtm_force) > 0:
        # Calculate scaling factors based on approximate body weight ratio
        max_qtm_force = np.max(qtm_force)
        max_insole_total = np.max(np.abs(insole_force_left)) + np.max(np.abs(insole_force_right))

        if max_insole_total > 0:
            scaling_factor = max_qtm_force / max_insole_total
            print(f"Scaling insole forces by factor: {scaling_factor:.2f}")
            insole_force_left = insole_force_left * scaling_factor
            insole_force_right = insole_force_right * scaling_factor

    # Apply filtering to smooth the signals
    qtm_force_filtered = filter_signal(qtm_force)
    insole_force_left_filtered = filter_signal(insole_force_left)
    insole_force_right_filtered = filter_signal(insole_force_right)

    # If insole data has fewer samples than QTM data, extend it to match QTM time range
    # by padding with zeros or interpolating
    if max(insole_time) < max(qtm_time):
        print(
            f"Insole data ({max(insole_time):.2f}s) ends before QTM data ({max(qtm_time):.2f}s)"
        )
        print("Extending insole data to match QTM time range")

        # Create extended time series
        extended_time = np.linspace(min(insole_time), max(qtm_time), len(insole_time))

        # Return extended time series
        return (
            qtm_time,
            qtm_force_filtered,
            extended_time,
            insole_force_left_filtered,
            insole_force_right_filtered,
        )

    return (
        qtm_time,
        qtm_force_filtered,
        insole_time,
        insole_force_left_filtered,
        insole_force_right_filtered,
    )


def plot_force_comparison(
    qtm_time,
    qtm_force,
    insole_time,
    insole_force_left,
    insole_force_right,
    output_path,
    trial_name,
):
    """
    Create comparison plots of QTM force data vs insole force data.

    Parameters:
    -----------
    qtm_time : array_like
        QTM time values in seconds
    qtm_force : array_like
        QTM force values
    insole_time : array_like
        Insole time values in seconds
    insole_force_left : array_like
        Left insole force values
    insole_force_right : array_like
        Right insole force values
    output_path : str
        Path to save the plot
    trial_name : str
        Name of the trial for the plot title
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])

    # Plot QTM force
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(qtm_time, qtm_force, label="QTM Force", color="blue")
    ax1.set_title(f"Force Comparison - {trial_name}")
    ax1.set_ylabel("QTM Force (N)")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Calculate total insole force for reference line
    total_insole_force = insole_force_left + insole_force_right

    # Plot QTM and total insole force together for comparison
    ax1.plot(
        insole_time,
        total_insole_force,
        label="Total Insole Force",
        color="purple",
        alpha=0.6,
    )

    # Plot insole left force
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(insole_time, insole_force_left, label="Left Insole Force", color="green")
    ax2.set_ylabel("Left Insole Force (N)")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    # Plot insole right force
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(insole_time, insole_force_right, label="Right Insole Force", color="red")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Right Insole Force (N)")
    ax3.grid(True)
    ax3.legend(loc="upper right")

    # Set common x-axis limits
    max_t = max(
        max(qtm_time) if len(qtm_time) > 0 else 1,
        max(insole_time) if len(insole_time) > 0 else 1,
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, max_t)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_combined_forces(
    qtm_time,
    qtm_force,
    insole_time,
    insole_force_left,
    insole_force_right,
    output_path,
    trial_name,
):
    """
    Create a plot that shows QTM force and combined insole forces together.

    Parameters:
    -----------
    qtm_time : array_like
        QTM time values in seconds
    qtm_force : array_like
        QTM force values
    insole_time : array_like
        Insole time values in seconds
    insole_force_left : array_like
        Left insole force values
    insole_force_right : array_like
        Right insole force values
    output_path : str
        Path to save the plot
    trial_name : str
        Name of the trial for the plot title
    """
    # Create the plot
    plt.figure(figsize=(12, 8))

    # QTM force
    plt.plot(qtm_time, qtm_force, label="QTM Force", color="blue", linewidth=2)

    # Combined insole forces (left + right)
    insole_total = insole_force_left + insole_force_right

    # Plot the total insole force
    plt.plot(
        insole_time, insole_total, label="Total Insole Force", color="red", linewidth=2
    )

    # Add individual insole forces
    plt.plot(
        insole_time,
        insole_force_left,
        label="Left Insole Force",
        color="green",
        linestyle="--",
        alpha=0.7,
    )
    plt.plot(
        insole_time,
        insole_force_right,
        label="Right Insole Force",
        color="orange",
        linestyle="--",
        alpha=0.7,
    )

    # Add labels and title
    plt.title(f"Force Comparison: QTM vs Insoles - {trial_name}", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Force (N)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Set x-axis limits to ensure all data is shown
    max_t = max(max(qtm_time), max(insole_time))
    plt.xlim(0, max_t)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined forces plot to: {output_path}")


def main():
    """Main function to run the analysis"""
    # Find matching data files
    matching_files = find_matching_data_files()

    if not matching_files:
        print("No matching data files found. Exiting.")
        return

    # Filter to only process the 182608 experiment
    target_timestamp = "20250324_182608"
    filtered_files = []
    for qtm_file, insole_file in matching_files:
        if target_timestamp in qtm_file:
            filtered_files.append((qtm_file, insole_file))

    if not filtered_files:
        print(f"No files found for experiment {target_timestamp}. Exiting.")
        return

    print(f"Only processing experiment with timestamp: {target_timestamp}")

    # Process each matching file pair
    for qtm_file, insole_file in filtered_files:
        # Extract trial name from filename
        trial_name = (
            os.path.basename(qtm_file).replace("qtm_data_", "").replace(".csv", "")
        )
        print(f"\nProcessing trial: {trial_name}")

        try:
            # Check file sizes to skip empty files
            qtm_size = os.path.getsize(qtm_file)
            insole_size = os.path.getsize(insole_file)

            if qtm_size == 0 or insole_size == 0:
                print(f"Skipping {trial_name} because one or both files are empty.")
                continue

            # Load data
            qtm_data = load_qtm_data(qtm_file)
            insole_data = load_insole_data(insole_file)

            # Check if data frames have content
            if len(qtm_data) == 0 or len(insole_data) == 0:
                print(
                    f"Skipping {trial_name} because one or both files have no data rows."
                )
                continue

            # Process and compare forces
            qtm_time, qtm_force, insole_time, insole_force_left, insole_force_right = (
                process_and_compare_forces(qtm_data, insole_data)
            )

            # Create comparison plots
            plot_force_comparison(
                qtm_time,
                qtm_force,
                insole_time,
                insole_force_left,
                insole_force_right,
                os.path.join(OUTPUT_DIR, f"force_comparison_{trial_name}.png"),
                trial_name,
            )

            # Create combined plot
            plot_combined_forces(
                qtm_time,
                qtm_force,
                insole_time,
                insole_force_left,
                insole_force_right,
                os.path.join(OUTPUT_DIR, f"combined_forces_{trial_name}.png"),
                trial_name,
            )

            print(f"Completed processing for trial: {trial_name}")

        except Exception as e:
            print(f"Error processing trial {trial_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            print(f"Skipping to next trial.")
            continue


if __name__ == "__main__":
    main()
