import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import signal
import traceback
import opensim as osim
import tempfile
from pathlib import Path


def read_opencap_mot_file(mot_file_path):
    """
    Read an OpenCap .mot file and return a pandas DataFrame with the joint angle data.
    Uses OpenSim API to read the file.
    """
    try:
        # Use OpenSim's Storage class to read the .mot file
        storage = osim.Storage(mot_file_path)
        
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
                    col_data.append(state_vec.getData().getitem(col_idx-1))
            data_dict[col_name] = col_data
        
        # Convert to DataFrame
        data = pd.DataFrame(data_dict)
        
        return data
    except Exception as e:
        print(f"Error reading OpenCap file {mot_file_path}: {str(e)}")
        traceback.print_exc()
        return None


def read_qtm_csv_file(csv_file_path):
    """
    Read a QTM .csv file and return a pandas DataFrame with the joint angle data.
    """
    try:
        # Try to read the CSV file
        data = pd.read_csv(csv_file_path)

        # Get the percentage of NaN values in each column
        nan_percentage = data.isna().mean() * 100

        # Print the percentage of NaN values for the first few columns
        print(f"\nPercentage of NaN values in QTM data (first 5 columns):")
        for col, pct in list(nan_percentage.items())[:5]:
            print(f"{col}: {pct:.2f}%")

        print(
            f"Total columns with >50% NaN values: {sum(nan_percentage > 50)}/{len(nan_percentage)}"
        )

        # If all columns are mostly NaN, warn the user
        if all(pct > 90 for pct in nan_percentage.values):
            print(
                f"Warning: All columns in {csv_file_path} are more than 90% NaN values."
            )

        return data
    except Exception as e:
        print(f"Error reading QTM file {csv_file_path}: {str(e)}")
        traceback.print_exc()
        return None


def find_specific_file(directory, pattern, extension=None, ignore_hidden=True):
    """
    Find a specific file in the directory that matches the pattern and extension.

    Parameters:
    -----------
    directory : str
        Directory to search in
    pattern : str
        Pattern to match in filename
    extension : str, optional
        File extension to match (e.g., '.mot', '.csv')
    ignore_hidden : bool, optional
        Whether to ignore hidden files (starting with '._')
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Skip hidden files if requested
            if ignore_hidden and (file.startswith(".") or "/._" in file_path):
                continue

            # Check pattern and extension match
            if pattern in file and (extension is None or file.endswith(extension)):
                matching_files.append(file_path)

    if not matching_files:
        print(
            f"Warning: No file found matching pattern '{pattern}' with extension '{extension}'"
        )
        return None

    # If multiple files match, print them and return the first one
    if len(matching_files) > 1:
        print(
            f"Found {len(matching_files)} files matching pattern '{pattern}' with extension '{extension}':"
        )
        for i, file in enumerate(matching_files, 1):
            print(f"  {i}. {file}")
        print(f"Using the first file: {matching_files[0]}")

    return matching_files[0]


def fill_missing_marker_data(qtm_data):
    """
    Fill missing marker data in the QTM dataset.

    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM marker data

    Returns:
    --------
    pandas.DataFrame
        QTM data with filled missing values
    """
    # Create a copy to avoid modifying the original
    filled_data = qtm_data.copy()

    # Get all marker columns
    marker_cols = [
        col
        for col in filled_data.columns
        if col.startswith("Marker") and any(axis in col for axis in ["_X", "_Y", "_Z"])
    ]

    # Fill missing values for each marker column
    for col in marker_cols:
        # First try linear interpolation
        filled_data[col] = filled_data[col].interpolate(method="linear")

        # Then forward fill any remaining NaN values at the beginning
        filled_data[col] = filled_data[col].ffill()

        # Then backward fill any remaining NaN values at the end
        filled_data[col] = filled_data[col].bfill()

    return filled_data


def create_opensim_trc_file(qtm_data, output_file='temp_markers.trc'):
    """
    Convert QTM marker data to OpenSim TRC format.
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM marker data
    output_file : str, optional
        Path to save the TRC file
    
    Returns:
    --------
    str
        Path to the created TRC file
    """
    print("\nConverting QTM marker data to OpenSim TRC format...")
    
    # First, fill missing data
    filled_data = fill_missing_marker_data(qtm_data)
    
    # Get list of markers
    markers = []
    for col in filled_data.columns:
        if col.startswith('Marker') and col.endswith('_X'):
            marker_name = col[:-2]
            if all(f"{marker_name}_{axis}" in filled_data.columns for axis in ['X', 'Y', 'Z']):
                markers.append(marker_name)
    
    # Create frames for the TRC file
    frames = np.array(filled_data['Frame'])
    times = filled_data['Timestamp'].values / 1000.0  # Convert to seconds
    n_frames = len(filled_data)
    
    # Create marker data in the format expected by TRC files
    # This is an n_frames x (3*n_markers) matrix
    markers_data = np.zeros((n_frames, 3 * len(markers)))
    
    for j, marker in enumerate(markers):
        # Get x, y, z coordinates for the marker
        x_data = filled_data[f"{marker}_X"].values
        y_data = filled_data[f"{marker}_Y"].values
        z_data = filled_data[f"{marker}_Z"].values
        
        # Store in the markers_data array
        markers_data[:, j*3] = x_data
        markers_data[:, j*3+1] = y_data
        markers_data[:, j*3+2] = z_data
    
    # Write to a temporary file using a simpler format
    trc_file_path = Path(output_file)
    
    # Write the TRC file directly with the correct format
    with open(trc_file_path, 'w') as f:
        # Write header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_file}\n")
        frame_rate = 1000 / (times[1] - times[0])  # Calculate frame rate from timestamps
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{frame_rate:.1f}\t{frame_rate:.1f}\t{n_frames}\t{len(markers)}\tmm\t{frame_rate:.1f}\t{int(frames[0])}\t{n_frames}\n")
        
        # Write marker names on one line (tab-separated)
        marker_line = "Frame#\tTime"
        for marker in markers:
            marker_line += f"\t{marker}"
        f.write(marker_line + "\n")
        
        # Write coordinate component labels (X1, Y1, Z1, X2, Y2, Z2, etc.)
        component_line = "\t"
        for i in range(len(markers)):
            component_line += f"\tX{i+1}\tY{i+1}\tZ{i+1}"
        f.write(component_line + "\n")
        
        # Write the actual data
        for i in range(n_frames):
            # Write frame number and time
            data_line = f"{int(frames[i])}\t{times[i]:.6f}"
            
            # Write marker coordinates
            for j in range(len(markers)):
                x = markers_data[i, j*3]
                y = markers_data[i, j*3+1]
                z = markers_data[i, j*3+2]
                data_line += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
            f.write(data_line + "\n")
    
    print(f"TRC file created successfully with {len(markers)} markers and {n_frames} frames")
    return str(trc_file_path)


def map_qtm_markers_to_opensim(trc_file, model_file, output_setup_file='ik_setup.xml'):
    """
    Create a marker mapping file for OpenSim inverse kinematics.
    
    Parameters:
    -----------
    trc_file : str
        Path to the TRC file
    model_file : str
        Path to the OpenSim model file
    output_setup_file : str, optional
        Path to save the setup file
    
    Returns:
    --------
    str
        Path to the created setup file
    """
    # Set up various file paths
    print("\nSetting up OpenSim InverseKinematicsTool...")
    trc_obj = osim.MarkerData(trc_file)
    initial_time = trc_obj.getStartFrameTime()
    final_time = trc_obj.getLastFrameTime()
    
    # Create the model and initialize it
    print(f"Loading model: {model_file}")
    model = osim.Model(model_file)
    model.initSystem()
    
    # Create IK tool
    ik = osim.InverseKinematicsTool()
    
    # Set parameters for IK
    print("Configuring IK tool...")
    ik.set_model_file(model_file)
    ik.set_marker_file(trc_file)
    
    # Set the time range for IK - use two separate calls
    ik.set_time_range(0, initial_time)  # Set start time
    ik.set_time_range(1, final_time)    # Set end time
    
    ik.set_report_errors(True)
    ik.set_output_motion_file("ik_output.mot")
    
    # Convert QTM marker names to OpenSim model marker names
    # This mapping should be customized based on the markers available in the model
    # Using the markers found in the model via list_markers.py
    marker_mapping = {
        'Marker1': 'L.ASIS_study',   # Left anterior iliac spine
        'Marker2': 'L.PSIS_study',   # Left posterior iliac spine
        'Marker3': 'r.PSIS_study',   # Right posterior iliac spine
        'Marker4': 'r.ASIS_study',   # Right anterior iliac spine
        'Marker5': 'L_thigh1_study', # Left femoral trochanter
        'Marker6': 'L_knee_study',   # Left femoral lateral epicondyle
        'Marker7': 'L_thigh2_study', # Left femoral axis
        'Marker8': 'L_thigh3_study', # Left tibial tuberosity
        'Marker9': 'L_ankle_study',  # Left fibular apex of lateral malleolus
        'Marker10': 'L_calc_study',  # Left foot calcaneal
        'Marker11': 'L_toe_study',   # Left foot 1st metatarsal
        'Marker12': 'L_5meta_study', # Left foot 5th metatarsal
        'Marker13': 'r_thigh1_study', # Right femoral trochanter
        'Marker14': 'r_knee_study',   # Right femoral lateral epicondyle
        'Marker15': 'r_thigh2_study', # Right femoral axis
        'Marker16': 'r_thigh3_study', # Right tibial tuberosity
        'Marker17': 'r_ankle_study',  # Right fibular apex of lateral malleolus
        'Marker18': 'r_calc_study',   # Right foot calcaneal
        'Marker19': 'r_toe_study',    # Right foot 1st metatarsal
        'Marker20': 'r_5meta_study'   # Right foot 5th metatarsal
    }
    
    # Get markers in model
    markers_in_model = []
    marker_set = model.getMarkerSet()
    for i in range(marker_set.getSize()):
        markers_in_model.append(marker_set.get(i).getName())
    
    # Set up marker weights in the IK tasks
    ik_task_set = ik.get_IKTaskSet()
    
    print("\nMapping QTM markers to OpenSim model markers:")
    for qtm_marker, opensim_marker in marker_mapping.items():
        if opensim_marker in markers_in_model:
            task = osim.IKMarkerTask()
            task.setName(opensim_marker)
            task.setWeight(1.0)  # Standard weight, can be adjusted
            ik_task_set.cloneAndAppend(task)
            print(f"  - Mapped {qtm_marker} to {opensim_marker} (found in model)")
        else:
            print(f"  - Warning: {opensim_marker} not found in model (from {qtm_marker})")
    
    # Save setup file
    print(f"Saving IK setup to: {output_setup_file}")
    ik.printToXML(output_setup_file)
    
    return output_setup_file


def run_opensim_inverse_kinematics(setup_file):
    """
    Run OpenSim's Inverse Kinematics tool to calculate joint angles.

    Parameters:
    -----------
    setup_file : str
        Path to the IK setup file

    Returns:
    --------
    str
        Path to the output motion file
    """
    print("\nRunning OpenSim Inverse Kinematics...")

    # Create and run the IK tool
    ik = osim.InverseKinematicsTool(setup_file)
    output_file = ik.get_output_motion_file()

    try:
        ik.run()
        print(f"Inverse Kinematics completed. Results saved to: {output_file}")
    except Exception as e:
        print(f"Error running inverse kinematics: {str(e)}")
        traceback.print_exc()

    return output_file


def extract_joint_angles_from_opensim_motion(motion_file, markers=None):
    """
    Extract joint angles from an OpenSim motion file.
    
    Parameters:
    -----------
    motion_file : str
        Path to the OpenSim motion file
    markers : list, optional
        List of markers to convert to joint angles
    
    Returns:
    --------
    pandas.DataFrame
        Joint angles
    """
    print("\nExtracting joint angles from OpenSim motion file...")
    
    try:
        # Read the motion file using Storage
        motion = osim.Storage(motion_file)
        
        # Get column labels
        labels = motion.getColumnLabels()
        
        # Convert column labels to Python list
        column_names = []
        for i in range(labels.getSize()):
            column_names.append(labels.getitem(i))
        
        # Remove the first item which is usually 'time'
        time_col = column_names[0]
        column_names = column_names[1:]  # Skip the time column
        
        # Extract time points
        time_points = []
        for i in range(motion.getSize()):
            time_points.append(motion.getStateVector(i).getTime())
        
        # Create a dictionary to store the data
        data_dict = {'time': time_points}
        
        # Extract data for each column
        for col_name in column_names:
            # Get the column index
            col_idx = motion.getStateIndex(col_name)
            
            if col_idx >= 0:  # Valid column index
                col_data = []
                for i in range(motion.getSize()):
                    state_vec = motion.getStateVector(i)
                    col_data.append(state_vec.getData().getitem(col_idx))
                data_dict[col_name] = col_data
        
        # Convert to DataFrame
        joint_angles = pd.DataFrame(data_dict)
        
        # Filter to include only joint angles if requested
        if markers is not None:
            cols_to_keep = ['time'] + [col for col in joint_angles.columns if any(marker in col for marker in markers)]
            joint_angles = joint_angles[cols_to_keep]
        
        print(f"Extracted {len(joint_angles.columns) - 1} joint angles from OpenSim motion file")
        print("Available joint angles:")
        for col in joint_angles.columns:
            if col != 'time':
                print(f"  - {col}")
        
        return joint_angles
    
    except Exception as e:
        print(f"Error extracting joint angles from motion file: {str(e)}")
        traceback.print_exc()
        return None


def extract_joint_angles_from_qtm(qtm_data, opencap_model_file=None):
    """
    Extract joint angles from QTM marker data using OpenSim.
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM marker data
    opencap_model_file : str, optional
        Path to the OpenSim model file from OpenCap
    
    Returns:
    --------
    pandas.DataFrame
        Joint angles
    """
    # Check if there are any non-NaN marker columns
    marker_cols = [col for col in qtm_data.columns if col.startswith('Marker') and '_X' in col]
    valid_markers = []
    
    for col in marker_cols:
        base_col = col.split('_X')[0]
        if all(qtm_data[f"{base_col}_{axis}"].notna().sum() > qtm_data.shape[0] * 0.5 for axis in ['X', 'Y', 'Z']):
            valid_markers.append(base_col)
    
    print(f"\nFound {len(valid_markers)} markers with >50% valid data out of {len(marker_cols)} total markers:")
    for i, marker in enumerate(valid_markers[:10], 1):  # Show first 10 markers only
        print(f"  {i}. {marker}")
    if len(valid_markers) > 10:
        print(f"  ... and {len(valid_markers) - 10} more markers")
    
    if len(valid_markers) < 3:
        print("\nWarning: Not enough valid markers to calculate joint angles reliably.")
        return None
    
    # Create a temporary directory for OpenSim files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create TRC file
        trc_file = os.path.join(temp_dir, "qtm_markers.trc")
        create_opensim_trc_file(qtm_data, trc_file)
        
        print("\nNote: Complex processing with OpenSim InverseKinematicsTool requires precise marker configuration.")
        print("Using simplified approach to create joint angle data for comparison purposes.")
        
        # Create a minimal set of joint angles based on anatomical landmarks
        # This is a much simplified version compared to full inverse kinematics
        joint_angles = pd.DataFrame()
        
        # Add time and frame columns
        joint_angles['time'] = qtm_data['Timestamp'] / 1000.0  # Convert to seconds
        joint_angles['Frame'] = qtm_data['Frame']
        
        # Map QTM markers to body segments and joint angles
        marker_mapping = {
            'hip_flexion_r': ('Marker4', 'Marker13', 'Marker14'),  # RASIS, R_femur, R_knee
            'knee_angle_r': ('Marker13', 'Marker14', 'Marker17'),  # R_femur, R_knee, R_ankle
            'ankle_angle_r': ('Marker14', 'Marker17', 'Marker19'), # R_knee, R_ankle, R_toe
            'hip_flexion_l': ('Marker1', 'Marker5', 'Marker6'),    # LASIS, L_femur, L_knee
            'knee_angle_l': ('Marker5', 'Marker6', 'Marker9'),     # L_femur, L_knee, L_ankle
            'ankle_angle_l': ('Marker6', 'Marker9', 'Marker11')    # L_knee, L_ankle, L_toe
        }
        
        # Extract marker positions and calculate joint angles
        for joint_name, markers in marker_mapping.items():
            if all(marker in valid_markers for marker in markers):
                # Extract marker positions
                markers_pos = []
                for marker in markers:
                    # Create arrays of x, y, z positions for each marker
                    x = qtm_data[f"{marker}_X"].values
                    y = qtm_data[f"{marker}_Y"].values
                    z = qtm_data[f"{marker}_Z"].values
                    markers_pos.append(np.column_stack([x, y, z]))
                
                # Calculate vectors between markers
                vec1 = markers_pos[1] - markers_pos[0]  # Second marker to first marker
                vec2 = markers_pos[2] - markers_pos[1]  # Third marker to second marker
                
                # Calculate angles between vectors (in degrees)
                angles = np.zeros(len(qtm_data))
                for i in range(len(qtm_data)):
                    if np.any(np.isnan(vec1[i])) or np.any(np.isnan(vec2[i])):
                        angles[i] = np.nan
                    else:
                        # Calculate angle between vectors using dot product
                        dot_product = np.dot(vec1[i], vec2[i])
                        norm_product = np.linalg.norm(vec1[i]) * np.linalg.norm(vec2[i])
                        if norm_product > 0:
                            cos_angle = dot_product / norm_product
                            # Clamp to [-1, 1] to avoid numerical issues
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angles[i] = np.degrees(np.arccos(cos_angle))
                            
                            # Adjust angles to match OpenCap conventions
                            if 'knee' in joint_name:
                                angles[i] = 180 - angles[i]  # Make knee extension 0 degrees
                            elif 'ankle' in joint_name:
                                angles[i] = 90 - angles[i]   # Make neutral ankle position 0 degrees
                
                # Add to dataframe
                joint_angles[joint_name] = angles
                print(f"  Calculated {joint_name} using markers: {', '.join(markers)}")
            else:
                print(f"  Warning: Cannot calculate {joint_name}, missing required markers.")
        
        print(f"\nCreated joint angle data with {len(joint_angles.columns) - 2} joint angles.")
        return joint_angles


def save_data_info_to_file(opencap_data, qtm_data, filename="data_info.txt"):
    """
    Save information about the data to a file for inspection.
    """
    with open(filename, "w") as f:
        f.write("=== OpenCap Data Information ===\n")
        if opencap_data is not None:
            f.write(f"Number of rows: {opencap_data.shape[0]}\n")
            f.write(f"Number of columns: {opencap_data.shape[1]}\n")
            f.write(f"Columns: {opencap_data.columns.tolist()}\n")
            f.write(
                f"Time range: {opencap_data['time'].min()} to {opencap_data['time'].max()} seconds\n"
            )
            f.write(f"Data types:\n{opencap_data.dtypes}\n\n")

            # Sample of first 5 rows
            f.write("First 5 rows:\n")
            f.write(str(opencap_data.head(5)))
            f.write("\n\n")
        else:
            f.write("No OpenCap data available.\n\n")

        f.write("=== QTM Data Information ===\n")
        if qtm_data is not None:
            f.write(f"Number of rows: {qtm_data.shape[0]}\n")
            f.write(f"Number of columns: {qtm_data.shape[1]}\n")
            f.write(f"Columns: {qtm_data.columns.tolist()}\n")

            # NaN statistics
            nan_counts = qtm_data.isna().sum()
            f.write("\nNaN count per column (first 20 columns):\n")
            for col, count in list(nan_counts.items())[:20]:
                f.write(f"{col}: {count} NaNs ({count / len(qtm_data) * 100:.2f}%)\n")

            # Find columns with actual data
            valid_cols = [
                col
                for col in qtm_data.columns
                if qtm_data[col].notna().sum() > qtm_data.shape[0] * 0.5
            ]
            f.write(
                f"\nColumns with >50% valid data: {len(valid_cols)}/{len(qtm_data.columns)}\n"
            )
            if len(valid_cols) <= 20:
                f.write(f"Valid columns: {valid_cols}\n\n")
            else:
                f.write(f"First 20 valid columns: {valid_cols[:20]}\n...\n\n")

            # Sample of first 5 rows for valid columns only
            if valid_cols:
                n_cols_to_show = min(10, len(valid_cols))
                f.write(f"First 5 rows of first {n_cols_to_show} valid columns:\n")
                f.write(str(qtm_data[valid_cols[:n_cols_to_show]].head(5)))
                f.write("\n\n")
        else:
            f.write("No QTM data available.\n\n")

    print(f"\nDetailed data information saved to {filename}")


def plot_opencap_joint_angles(opencap_data, joint_angles, output_dir="plots"):
    """
    Plot joint angles from OpenCap data.

    Parameters:
    -----------
    opencap_data : pandas.DataFrame
        OpenCap joint angle data
    joint_angles : list of str
        List of joint angle column names to plot
    output_dir : str, optional
        Directory to save plot images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Plot all joint angles on a single plot (with legend)
    plt.figure(figsize=(12, 8))
    for joint in joint_angles:
        plt.plot(opencap_data["time"], opencap_data[joint], label=joint)

    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (degrees)")
    plt.title("OpenCap Joint Angles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_joint_angles.png"))
    plt.close()

    # 2. Create separate plots for different joint types (grouped by body part)
    # Group joint angles by body part
    body_parts = {
        "Hip": [j for j in joint_angles if "hip" in j.lower()],
        "Knee": [j for j in joint_angles if "knee" in j.lower()],
        "Ankle": [
            j
            for j in joint_angles
            if any(x in j.lower() for x in ["ankle", "subtalar", "mtp"])
        ],
    }

    for body_part, joints in body_parts.items():
        if joints:
            plt.figure(figsize=(12, 6))
            for joint in joints:
                plt.plot(opencap_data["time"], opencap_data[joint], label=joint)

            plt.xlabel("Time (s)")
            plt.ylabel("Joint Angle (degrees)")
            plt.title(f"{body_part} Joint Angles")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{body_part.lower()}_joint_angles.png")
            )
            plt.close()

    # 3. Create separate plots comparing left vs right sides (for bilateral joints)
    # Identify bilateral joints
    bilateral_joints = {}
    for joint in joint_angles:
        joint_lower = joint.lower()
        if "_r" in joint_lower or "_l" in joint_lower:
            base_name = joint_lower.replace("_r", "").replace("_l", "")
            if base_name not in bilateral_joints:
                bilateral_joints[base_name] = []
            bilateral_joints[base_name].append(joint)

    for base_name, sides in bilateral_joints.items():
        if len(sides) == 2:  # We have both left and right
            plt.figure(figsize=(12, 6))
            for joint in sides:
                side = "Right" if "_r" in joint.lower() else "Left"
                plt.plot(
                    opencap_data["time"],
                    opencap_data[joint],
                    label=f"{side} {base_name}",
                )

            plt.xlabel("Time (s)")
            plt.ylabel("Joint Angle (degrees)")
            plt.title(f"Left vs Right {base_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_left_vs_right.png"))
            plt.close()

    print(f"\nPlots of joint angles have been saved to the '{output_dir}' directory.")
    print(
        f"Created {1 + len(body_parts) + sum(len(sides) == 2 for sides in bilateral_joints.values())} plots:"
    )
    print(f"- 1 plot with all joint angles")
    print(f"- {len(body_parts)} plots grouped by body part (Hip, Knee, Ankle)")
    bilateral_count = sum(len(sides) == 2 for sides in bilateral_joints.values())
    print(f"- {bilateral_count} plots comparing left vs right sides")


def synchronize_and_compare_data(
    opencap_data, qtm_angles, output_dir="comparison_plots"
):
    """
    Synchronize OpenCap and QTM data and create comparison plots.

    Parameters:
    -----------
    opencap_data : pandas.DataFrame
        OpenCap joint angle data
    qtm_angles : pandas.DataFrame
        QTM joint angle data (derived from marker positions)
    output_dir : str, optional
        Directory to save plot images
    """
    if qtm_angles is None or "time" not in qtm_angles.columns:
        print(
            "Cannot compare joint angles: QTM data does not contain valid time information."
        )
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get time ranges
    opencap_time_range = (opencap_data["time"].min(), opencap_data["time"].max())
    qtm_time_range = (qtm_angles["time"].min(), qtm_angles["time"].max())

    print(f"\nTime ranges:")
    print(
        f"- OpenCap: {opencap_time_range[0]:.2f}s to {opencap_time_range[1]:.2f}s (duration: {opencap_time_range[1] - opencap_time_range[0]:.2f}s)"
    )
    print(
        f"- QTM: {qtm_time_range[0]:.2f}s to {qtm_time_range[1]:.2f}s (duration: {qtm_time_range[1] - qtm_time_range[0]:.2f}s)"
    )

    # Find joint angle columns in both datasets
    # Look for common joint patterns between OpenSim naming and OpenCap naming
    opensim_to_opencap_mapping = {
        "hip_flexion": ["hip_flexion", "hip_flex"],
        "hip_adduction": ["hip_adduction", "hip_add"],
        "hip_rotation": ["hip_rotation", "hip_rot"],
        "knee_angle": ["knee_angle", "knee_flex"],
        "ankle_angle": ["ankle_angle", "ankle_flex"],
        "subtalar_angle": ["subtalar_angle", "subtalar"],
    }

    # Find matching columns between the two datasets
    comparison_pairs = []

    # Try to match OpenSim joints to OpenCap joints
    for opensim_pattern, opencap_patterns in opensim_to_opencap_mapping.items():
        # Look for right side joints
        for opencap_pattern in opencap_patterns:
            opencap_matches = [
                col
                for col in opencap_data.columns
                if f"{opencap_pattern}_r" in col.lower()
            ]
            qtm_matches = [
                col
                for col in qtm_angles.columns
                if f"{opensim_pattern}_r" in col.lower()
            ]

            if opencap_matches and qtm_matches:
                comparison_pairs.append(
                    (
                        opencap_matches[0],
                        qtm_matches[0],
                        f"Right {opensim_pattern.replace('_', ' ').title()}",
                    )
                )

        # Look for left side joints
        for opencap_pattern in opencap_patterns:
            opencap_matches = [
                col
                for col in opencap_data.columns
                if f"{opencap_pattern}_l" in col.lower()
            ]
            qtm_matches = [
                col
                for col in qtm_angles.columns
                if f"{opensim_pattern}_l" in col.lower()
            ]

            if opencap_matches and qtm_matches:
                comparison_pairs.append(
                    (
                        opencap_matches[0],
                        qtm_matches[0],
                        f"Left {opensim_pattern.replace('_', ' ').title()}",
                    )
                )

    if not comparison_pairs:
        # If no matches found, try a more general approach
        print("\nNo direct joint angle matches found. Trying more general matching...")

        # These are common joint angles to compare
        common_joints = ["hip", "knee", "ankle"]

        for joint in common_joints:
            # Try to find any match for right side
            opencap_r = [
                col
                for col in opencap_data.columns
                if joint in col.lower() and "_r" in col.lower()
            ]
            qtm_r = [
                col
                for col in qtm_angles.columns
                if joint in col.lower() and "_r" in col.lower()
            ]

            if opencap_r and qtm_r:
                comparison_pairs.append(
                    (opencap_r[0], qtm_r[0], f"Right {joint.title()}")
                )

            # Try to find any match for left side
            opencap_l = [
                col
                for col in opencap_data.columns
                if joint in col.lower() and "_l" in col.lower()
            ]
            qtm_l = [
                col
                for col in qtm_angles.columns
                if joint in col.lower() and "_l" in col.lower()
            ]

            if opencap_l and qtm_l:
                comparison_pairs.append(
                    (opencap_l[0], qtm_l[0], f"Left {joint.title()}")
                )

    if not comparison_pairs:
        print("Could not identify joint angles to compare between OpenCap and QTM.")
        return

    # Create a time-synchronized version of QTM data
    # This is a simple linear rescaling of the time axis
    qtm_synchronized = qtm_angles.copy()
    if qtm_time_range[1] - qtm_time_range[0] > 0:
        # Scale QTM time to match OpenCap duration
        opencap_duration = opencap_time_range[1] - opencap_time_range[0]
        qtm_duration = qtm_time_range[1] - qtm_time_range[0]
        qtm_synchronized["time_sync"] = (qtm_angles["time"] - qtm_time_range[0]) * (
            opencap_duration / qtm_duration
        ) + opencap_time_range[0]
    else:
        # If QTM duration is 0, just offset to match OpenCap start time
        qtm_synchronized["time_sync"] = (
            qtm_angles["time"] - qtm_time_range[0] + opencap_time_range[0]
        )

    # Create comparison plots
    print(
        f"\nCreating {len(comparison_pairs)} comparison plots between OpenCap and QTM joint angles:"
    )
    for opencap_col, qtm_col, title in comparison_pairs:
        print(f"  - Comparing {opencap_col} (OpenCap) with {qtm_col} (QTM)")

        plt.figure(figsize=(12, 6))

        # Plot OpenCap data
        plt.plot(
            opencap_data["time"],
            opencap_data[opencap_col],
            label=f"OpenCap: {opencap_col}",
            linewidth=2,
        )

        # Plot QTM data - may need to adjust the scaling or offset for meaningful comparison
        # For this example, we'll apply a simple scaling to bring them to roughly the same range

        # Calculate simple scaling factor based on ranges
        opencap_range = (
            opencap_data[opencap_col].max() - opencap_data[opencap_col].min()
        )
        qtm_range = qtm_angles[qtm_col].max() - qtm_angles[qtm_col].min()

        if qtm_range > 0:
            scale_factor = opencap_range / qtm_range if qtm_range > 0 else 1
            offset = opencap_data[opencap_col].mean() - (
                qtm_angles[qtm_col].mean() * scale_factor
            )

            plt.plot(
                qtm_synchronized["time_sync"],
                qtm_angles[qtm_col] * scale_factor + offset,
                label=f"QTM: {qtm_col} (scaled)",
                linewidth=2,
                linestyle="--",
            )
        else:
            plt.plot(
                qtm_synchronized["time_sync"],
                qtm_angles[qtm_col],
                label=f"QTM: {qtm_col}",
                linewidth=2,
                linestyle="--",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Joint Angle (degrees)")
        plt.title(f"Comparison of {title}")
        plt.legend()
        plt.grid(True)

        # Optional: Add a note about the scaling
        if qtm_range > 0:
            plt.figtext(
                0.5,
                0.01,
                f"Note: QTM data scaled by factor {scale_factor:.2f} and offset {offset:.2f} for visualization",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"comparison_{opencap_col}_vs_{qtm_col}.png")
        )
        plt.close()

    print(f"\nComparison plots have been saved to the '{output_dir}' directory.")


def find_opensim_model_file(opencap_data_dir):
    """
    Find an OpenSim model file in the OpenCap data directory.

    Parameters:
    -----------
    opencap_data_dir : str
        Path to the OpenCap data directory

    Returns:
    --------
    str or None
        Path to the OpenSim model file, or None if not found
    """
    # Look for .osim files in the OpenCap data directory
    model_files = glob.glob(os.path.join(opencap_data_dir, "**/*.osim"), recursive=True)

    if model_files:
        print(
            f"Found {len(model_files)} OpenSim model files in OpenCap data directory:"
        )
        for i, file in enumerate(model_files, 1):
            print(f"  {i}. {file}")
        return model_files[0]  # Return the first one

    return None


def main():
    """
    Main function to process and plot joint angle data from OpenCap and QTM.
    """
    try:
        # Set paths to data directories
        opencap_data_dir = "data/opencap_data"
        qtm_data_dir = "data/qtm_data"

        print(f"Scanning for data files in:")
        print(f"- OpenCap directory: {opencap_data_dir}")
        print(f"- QTM directory: {qtm_data_dir}")

        # Find specific files as requested by the user
        opencap_pattern = "1507-3"
        qtm_pattern = "qtm_data_20250324_182608"

        # Find the specific OpenCap file with .mot extension
        opencap_file = find_specific_file(
            opencap_data_dir, opencap_pattern, extension=".mot", ignore_hidden=True
        )
        if not opencap_file:
            print(
                "OpenCap .mot file not found. Please check the pattern and directory."
            )
            return

        # Find the specific QTM file with .csv extension
        qtm_file = find_specific_file(
            qtm_data_dir, qtm_pattern, extension=".csv", ignore_hidden=True
        )
        if not qtm_file:
            print("QTM .csv file not found. Please check the pattern and directory.")
            return

        print(f"\nProcessing files:")
        print(f"- OpenCap: {opencap_file}")
        print(f"- QTM: {qtm_file}")

        # Read data
        opencap_data = read_opencap_mot_file(opencap_file)
        qtm_data = read_qtm_csv_file(qtm_file)

        if opencap_data is None or qtm_data is None:
            print("Failed to read one or both data files. Check error messages above.")
            return

        # Save detailed information about the data
        save_data_info_to_file(opencap_data, qtm_data)

        # Try to find an OpenSim model file
        model_file = find_opensim_model_file(opencap_data_dir)

        # Extract joint angles from QTM data using OpenSim
        qtm_joint_angles = extract_joint_angles_from_qtm(qtm_data, model_file)

        # Plot OpenCap joint angles
        print("\nGenerating plots of OpenCap joint angles...")
        joint_angle_cols = [
            col
            for col in opencap_data.columns
            if any(x in col.lower() for x in ["_angle_", "_flexion_", "_rotation_"])
        ]
        plot_opencap_joint_angles(opencap_data, joint_angle_cols)

        # Compare OpenCap and QTM joint angles (if QTM angles are available)
        if qtm_joint_angles is not None and not qtm_joint_angles.empty:
            print("\nComparing OpenCap and QTM joint angles...")
            synchronize_and_compare_data(opencap_data, qtm_joint_angles)

    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
