"""
Joint angle processing functions
"""

import numpy as np
import pandas as pd


def extract_opencap_joint_angles(opencap_data):
    """
    Extract all joint angles from OpenCap data
    
    Parameters:
    -----------
    opencap_data : pandas.DataFrame
        OpenCap data
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing all joint angle data, or None if no angles found
    """
    # Find all angle/kinematic columns
    # Common angle naming patterns in OpenCap data
    angle_patterns = [
        "angle", "flexion", "rotation", "adduction", "abduction", 
        "eversion", "inversion", "dorsiflexion", "plantarflexion",
        "tilt", "list", "twist", "elevation", "depression"
    ]
    
    # Get all columns that contain any of these patterns
    angle_cols = []
    for pattern in angle_patterns:
        angle_cols.extend([col for col in opencap_data.columns if pattern in col.lower()])
    
    # Remove duplicates while preserving order
    angle_cols = list(dict.fromkeys(angle_cols))
    
    if not angle_cols:
        print("No joint angle columns found in OpenCap data")
        return None

    print(f"Found {len(angle_cols)} joint angle columns in OpenCap data")
    
    # Return subset with time and all angle columns
    return opencap_data[["time"] + angle_cols]


def extract_qtm_joint_angles(qtm_data):
    """
    Extract all joint angles from QTM data
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all QTM joint angle data
    """
    # First try to find any named joint angle columns in QTM data
    # Patterns that might indicate joint angle columns
    angle_patterns = [
        "angle", "flexion", "rotation", "adduction", "abduction", 
        "hip", "knee", "ankle", "pelvis", "spine", "shoulder", "elbow", "wrist",
        "foot", "trunk", "thorax"
    ]
    
    # Get all columns that might contain joint angles
    angle_cols = []
    for pattern in angle_patterns:
        matching_cols = [col for col in qtm_data.columns if pattern in col.lower()]
        angle_cols.extend(matching_cols)
    
    # Remove duplicates while preserving order
    angle_cols = list(dict.fromkeys(angle_cols))
    
    # If we didn't find any columns but we have marker data, calculate angles
    if not angle_cols or all("marker" in col.lower() for col in angle_cols):
        print("No named joint angle columns found in QTM data")
        
        # Calculate joint angles from marker positions
        joint_angle_data = calculate_joint_angles_from_markers(qtm_data)
        
        # Return the calculated angles
        return joint_angle_data
    else:
        print(f"Found {len(angle_cols)} joint angle columns in QTM data")
        
        # Return subset with time and all angle columns
        return qtm_data[["TimeSeconds"] + angle_cols]


def calculate_joint_angles_from_markers(qtm_data):
    """
    Calculate multiple joint angles from marker positions in QTM data
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM data with marker positions
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all calculated joint angles
    """
    print("Calculating joint angles from QTM marker positions")

    # Initialize result dataframe
    joint_angle_data = pd.DataFrame()
    joint_angle_data["TimeSeconds"] = qtm_data["TimeSeconds"]
    
    # Define marker setups for different joint angles
    marker_setups = {
        # Knee angles
        "knee_angle_r": {
            "proximal": [col for col in qtm_data.columns if "Marker13" in col and "_X" in col],  # Thigh
            "middle": [col for col in qtm_data.columns if "Marker14" in col and "_X" in col],    # Knee
            "distal": [col for col in qtm_data.columns if "Marker17" in col and "_X" in col]     # Ankle
        },
        "knee_angle_l": {
            "proximal": [col for col in qtm_data.columns if "Marker5" in col and "_X" in col],   # Thigh
            "middle": [col for col in qtm_data.columns if "Marker6" in col and "_X" in col],     # Knee
            "distal": [col for col in qtm_data.columns if "Marker9" in col and "_X" in col]      # Ankle
        },
        # Hip angles (flexion)
        "hip_flexion_r": {
            "proximal": [col for col in qtm_data.columns if "Marker4" in col and "_X" in col],   # Pelvis
            "middle": [col for col in qtm_data.columns if "Marker13" in col and "_X" in col],    # Hip
            "distal": [col for col in qtm_data.columns if "Marker14" in col and "_X" in col]     # Knee
        },
        "hip_flexion_l": {
            "proximal": [col for col in qtm_data.columns if "Marker1" in col and "_X" in col],   # Pelvis
            "middle": [col for col in qtm_data.columns if "Marker5" in col and "_X" in col],     # Hip
            "distal": [col for col in qtm_data.columns if "Marker6" in col and "_X" in col]      # Knee
        },
        # Ankle angles
        "ankle_angle_r": {
            "proximal": [col for col in qtm_data.columns if "Marker14" in col and "_X" in col],  # Knee
            "middle": [col for col in qtm_data.columns if "Marker17" in col and "_X" in col],    # Ankle
            "distal": [col for col in qtm_data.columns if "Marker19" in col and "_X" in col]     # Toe
        },
        "ankle_angle_l": {
            "proximal": [col for col in qtm_data.columns if "Marker6" in col and "_X" in col],   # Knee
            "middle": [col for col in qtm_data.columns if "Marker9" in col and "_X" in col],     # Ankle
            "distal": [col for col in qtm_data.columns if "Marker11" in col and "_X" in col]     # Toe
        }
    }
    
    # Calculate each joint angle if the required markers are available
    for joint_name, markers in marker_setups.items():
        if all(len(markers[k]) > 0 for k in ["proximal", "middle", "distal"]):
            print(f"Calculating {joint_name} from markers")
            joint_angle_data[joint_name] = _calculate_angle_from_markers(
                qtm_data, 
                markers["proximal"][0].replace("_X", ""),
                markers["middle"][0].replace("_X", ""),
                markers["distal"][0].replace("_X", "")
            )
            
            # Apply angle conversion based on joint type
            if "knee" in joint_name:
                # Knee convention: 0° = full extension, 180° = full flexion
                joint_angle_data[joint_name] = joint_angle_data[joint_name]
            elif "ankle" in joint_name:
                # Ankle convention: 0° = neutral, positive = dorsiflexion, negative = plantarflexion
                joint_angle_data[joint_name] = 90 - joint_angle_data[joint_name]

    return joint_angle_data


def _calculate_angle_from_markers(qtm_data, thigh_marker, knee_marker, ankle_marker):
    """
    Helper function to calculate joint angle from three markers
    
    Parameters:
    -----------
    qtm_data : pandas.DataFrame
        QTM data with marker positions
    thigh_marker : str
        Base name of thigh marker (without _X, _Y, _Z)
    knee_marker : str
        Base name of knee marker
    ankle_marker : str
        Base name of ankle marker
        
    Returns:
    --------
    numpy.ndarray
        Array of calculated joint angles
    """
    # Create arrays of x, y, z positions for each marker
    thigh_pos = np.column_stack(
        [
            qtm_data[f"{thigh_marker}_X"].values,
            qtm_data[f"{thigh_marker}_Y"].values,
            qtm_data[f"{thigh_marker}_Z"].values,
        ]
    )

    knee_pos = np.column_stack(
        [
            qtm_data[f"{knee_marker}_X"].values,
            qtm_data[f"{knee_marker}_Y"].values,
            qtm_data[f"{knee_marker}_Z"].values,
        ]
    )

    ankle_pos = np.column_stack(
        [
            qtm_data[f"{ankle_marker}_X"].values,
            qtm_data[f"{ankle_marker}_Y"].values,
            qtm_data[f"{ankle_marker}_Z"].values,
        ]
    )

    # Calculate vectors between markers
    thigh_vec = knee_pos - thigh_pos  # Thigh to knee vector
    shank_vec = ankle_pos - knee_pos  # Knee to ankle vector

    # Calculate angles between vectors (in degrees)
    angles = np.zeros(len(qtm_data))
    for i in range(len(qtm_data)):
        if np.any(np.isnan(thigh_vec[i])) or np.any(np.isnan(shank_vec[i])):
            angles[i] = np.nan
        else:
            # Calculate angle between vectors using dot product
            dot_product = np.dot(thigh_vec[i], shank_vec[i])
            norm_product = np.linalg.norm(thigh_vec[i]) * np.linalg.norm(
                shank_vec[i]
            )
            if norm_product > 0:
                cos_angle = dot_product / norm_product
                # Clamp to [-1, 1] to avoid numerical issues
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles[i] = np.degrees(np.arccos(cos_angle))

                # Adjust angle to match knee flexion convention (180 - angle)
                angles[i] = 180 - angles[i]
    
    return angles