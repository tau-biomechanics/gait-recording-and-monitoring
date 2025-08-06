"""
ML dataset generation module for biomechanical data
"""

import os
import numpy as np
import pandas as pd
from ..utils.signal_processing import filter_signal
try:
    from PyQt5.QtWidgets import QMessageBox, QFileDialog
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")
from ..ml import JointMomentCNN
import matplotlib.pyplot as plt


def generate_ml_dataset(player_instance, time_range=None):
    """
    Generate a dataset for ML training using OpenGo force and OpenCap kinematics as input,
    and QTM-based joint moments as targets.
    
    Parameters
    ----------
    player_instance : QtSynchronizedVideoPlayer
        The player instance containing the data
    time_range : tuple, optional
        Time range (start_time, end_time) to use for the dataset
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the generated dataset
    """
    try:
        # Get the time range
        if time_range is None:
            start_time = 0.0
            end_time = player_instance.duration
        else:
            start_time, end_time = time_range
        
        print(f"Generating ML dataset from time {start_time:.2f}s to {end_time:.2f}s")
        
        # Extract indices for each data stream
        indices = {}
        idx = 0
        
        if player_instance.insole_data is not None:
            indices['insole'] = idx
            idx += 1
        
        if player_instance.qtm_force_data is not None:
            indices['qtm_force'] = idx
            idx += 1
            
        if player_instance.opencap_joint_data is not None:
            indices['opencap'] = idx
            idx += 1
            
        if player_instance.qtm_joint_data is not None:
            indices['qtm_joint'] = idx
            idx += 1
        
        # Check if we have all required data streams
        required_streams = ['insole', 'qtm_force', 'opencap', 'qtm_joint']
        for stream in required_streams:
            if stream not in indices:
                raise ValueError(f"Missing required data stream: {stream}")
        
        # Get synchronized data frames
        insole_sync = player_instance.sync_frames[indices['insole']]
        qtm_force_sync = player_instance.sync_frames[indices['qtm_force']]
        opencap_sync = player_instance.sync_frames[indices['opencap']]
        qtm_joint_sync = player_instance.sync_frames[indices['qtm_joint']]
        
        # Apply offsets to time values
        insole_time = insole_sync['time'] + player_instance.insole_offset
        qtm_force_time = qtm_force_sync['time'] + player_instance.qtm_force_offset
        opencap_time = opencap_sync['time'] + player_instance.opencap_joint_offset
        qtm_joint_time = qtm_joint_sync['time'] + player_instance.qtm_joint_offset
        
        # Filter to the desired time range
        insole_mask = (insole_time >= start_time) & (insole_time <= end_time)
        qtm_force_mask = (qtm_force_time >= start_time) & (qtm_force_time <= end_time)
        opencap_mask = (opencap_time >= start_time) & (opencap_time <= end_time)
        qtm_joint_mask = (qtm_joint_time >= start_time) & (qtm_joint_time <= end_time)
        
        # Apply masks
        insole_sync_filtered = insole_sync[insole_mask].copy()
        insole_time_filtered = insole_time[insole_mask]
        
        qtm_force_sync_filtered = qtm_force_sync[qtm_force_mask].copy()
        qtm_force_time_filtered = qtm_force_time[qtm_force_mask]
        
        opencap_sync_filtered = opencap_sync[opencap_mask].copy()
        opencap_time_filtered = opencap_time[opencap_mask]
        
        qtm_joint_sync_filtered = qtm_joint_sync[qtm_joint_mask].copy()
        qtm_joint_time_filtered = qtm_joint_time[qtm_joint_mask]
        
        # Check if we have data in the selected range
        if (len(insole_sync_filtered) == 0 or len(qtm_force_sync_filtered) == 0 or
            len(opencap_sync_filtered) == 0 or len(qtm_joint_sync_filtered) == 0):
            raise ValueError("No data available in the selected time range")
        
        # Create target variables - calculate joint moments using QTM data
        if player_instance.joint_moment_params:
            moments_data = {}
            
            print(f"Available joint moment parameters: {player_instance.joint_moment_params}")
            print(f"Available QTM parameters: {player_instance.qtm_params}")
            
            for joint_moment in player_instance.joint_moment_params:
                # Find the appropriate joint angle parameter based on the moment
                if 'Ankle' in joint_moment:
                    angle_params = [p for p in player_instance.qtm_params if 'ankle' in p.lower()]
                    joint_type = "Ankle"
                elif 'Knee' in joint_moment:
                    angle_params = [p for p in player_instance.qtm_params if 'knee' in p.lower()]
                    joint_type = "Knee"
                elif 'Hip' in joint_moment:
                    angle_params = [p for p in player_instance.qtm_params if 'hip' in p.lower()]
                    joint_type = "Hip"
                else:
                    print(f"Skipping unknown joint type in {joint_moment}")
                    continue
                
                print(f"For {joint_moment}, found matching angle parameters: {angle_params}")
                
                if not angle_params:
                    print(f"Warning: No matching angle parameters found for {joint_moment}")
                    continue
                    
                angle_param = angle_params[0]
                
                # Get force data
                forces = qtm_force_sync_filtered['Force'].values
                
                # Get angle data
                angles = qtm_joint_sync_filtered[angle_param].values
                
                print(f"Force data shape: {forces.shape}, non-NaN values: {np.sum(~np.isnan(forces))}")
                print(f"Angle data shape: {angles.shape}, non-NaN values: {np.sum(~np.isnan(angles))}")
                
                # Adjust for knee angles if needed
                if 'knee' in angle_param.lower():
                    angles = -angles + 180
                
                # Check data lengths and resample if necessary
                if len(forces) != len(angles):
                    # Resample to the shorter length
                    min_len = min(len(forces), len(angles))
                    forces = forces[:min_len]
                    angles = angles[:min_len]
                
                # Calculate joint moments
                from ..gui.qt_player import calculate_joint_moments
                try:
                    # Calculate moments using a more descriptive name
                    joint_type_lower = joint_type.lower()
                    
                    # Print details before calculation
                    print(f"Calculating {joint_type} moments with {len(forces)} force points and {len(angles)} angle points")
                    print(f"Force range: {np.nanmin(forces)}-{np.nanmax(forces)}")
                    print(f"Angle range: {np.nanmin(angles)}-{np.nanmax(angles)}")
                    
                    moments = calculate_joint_moments(forces, angles, player_instance.subject_mass)
                    print(f"Calculated moments: {len(moments)} values, non-NaN values: {np.sum(~np.isnan(moments))}")
                    
                    # Create a standardized name for the moment
                    moment_key = f"{joint_type_lower}_moment"
                    
                    # If calculation was successful, store results
                    if moments is not None and len(moments) > 0:
                        moments_data[moment_key] = moments
                        print(f"Stored moment data with key: {moment_key}")
                except Exception as e:
                    print(f"Error calculating {joint_type} moment: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    moments = np.array([])
        
        # Create common time points for resampling
        # Use the densest time series for the common time axis
        time_lengths = [
            len(insole_time_filtered),
            len(qtm_force_time_filtered),
            len(opencap_time_filtered),
            len(qtm_joint_time_filtered)
        ]
        
        densest_idx = np.argmax(time_lengths)
        
        if densest_idx == 0:
            common_time = insole_time_filtered
        elif densest_idx == 1:
            common_time = qtm_force_time_filtered
        elif densest_idx == 2:
            common_time = opencap_time_filtered
        else:
            common_time = qtm_joint_time_filtered
        
        # Create the dataset DataFrame
        dataset = pd.DataFrame({'time': common_time})
        
        # Add input features: OpenGo force
        if 'Left_Force' in insole_sync_filtered.columns and 'Right_Force' in insole_sync_filtered.columns:
            # Resample to common time points
            left_force_resampled = np.interp(
                common_time,
                insole_time_filtered,
                filter_signal(insole_sync_filtered['Left_Force'].values)
            )
            
            right_force_resampled = np.interp(
                common_time,
                insole_time_filtered,
                filter_signal(insole_sync_filtered['Right_Force'].values)
            )
            
            # Add to dataset
            dataset['insole_left_force'] = left_force_resampled
            dataset['insole_right_force'] = right_force_resampled
            dataset['insole_total_force'] = left_force_resampled + right_force_resampled
        
        # Add input features: OpenCap kinematics
        opencap_params = [col for col in opencap_sync_filtered.columns 
                         if col != 'time' and not col.startswith('Time')]
        
        for param in opencap_params:
            # Resample to common time points
            angles_resampled = np.interp(
                common_time,
                opencap_time_filtered,
                filter_signal(opencap_sync_filtered[param].values)
            )
            
            # Add to dataset
            dataset[f'opencap_{param}'] = angles_resampled
        
        # Add QTM force data
        if 'Force' in qtm_force_sync_filtered.columns:
            # Resample to common time points
            qtm_force_resampled = np.interp(
                common_time,
                qtm_force_time_filtered,
                filter_signal(qtm_force_sync_filtered['Force'].values)
            )
            
            # Add to dataset
            dataset['qtm_force'] = qtm_force_resampled
        
        # Add QTM joint angles
        qtm_params = [col for col in qtm_joint_sync_filtered.columns 
                     if col != 'time' and not col.startswith('Time')]
        
        for param in qtm_params:
            # Get angle data
            angles = qtm_joint_sync_filtered[param].values
            
            # Adjust for knee angles if needed
            if 'knee' in param.lower():
                angles = -angles + 180  # Invert direction to match OpenCap convention
            
            # Resample to common time points
            angles_resampled = np.interp(
                common_time,
                qtm_joint_time_filtered,
                filter_signal(angles)
            )
            
            # Add to dataset
            dataset[f'qtm_{param}'] = angles_resampled
        
        # Add target features: Joint moments
        for moment_name, moment_values in moments_data.items():
            # Resample to common time points if needed
            if len(moment_values) != len(common_time):
                # Create time array for moments
                moment_time = qtm_force_time_filtered[:len(moment_values)]
                
                # Resample
                moment_resampled = np.interp(
                    common_time,
                    moment_time,
                    filter_signal(moment_values)
                )
            else:
                moment_resampled = filter_signal(moment_values)
            
            # Add to dataset
            # Use a consistent naming convention for joint moments to ensure they're properly identified
            # Make sure "moment" is in lowercase for consistent filtering
            moment_column_name = f"qtm_{moment_name.lower()}"
            dataset[moment_column_name] = moment_resampled
            print(f"Added moment data to dataset: {moment_column_name} with {len(moment_resampled)} points")
        
        # Final check on the dataset
        if len(dataset) == 0:
            raise ValueError("Generated dataset is empty")
        
        # Report all columns for debugging
        print("Final dataset columns:")
        for col in dataset.columns:
            print(f"  - {col}")
            
        # Check specifically for moment columns
        moment_cols = [col for col in dataset.columns if 'moment' in col.lower()]
        print(f"Found {len(moment_cols)} moment columns: {moment_cols}")
        
        print(f"Generated ML dataset with {len(dataset)} rows and {len(dataset.columns)} columns")
        
        return dataset
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Show error dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f"Error generating ML dataset: {str(e)}")
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        return None


def show_ml_dataset_window(player_instance):
    """
    Generate ML dataset and show window for visualization and export
    
    Parameters
    ----------
    player_instance : QtSynchronizedVideoPlayer
        The player instance containing the data
    """
    try:
        # Get time range from UI
        start_time = float(player_instance.comparison_start_time.text())
        end_time = float(player_instance.comparison_end_time.text())
        
        # Validate time range
        start_time = max(0.0, start_time)
        end_time = min(player_instance.duration, end_time)
        
        if start_time >= end_time:
            end_time = start_time + 0.1
            end_time = min(player_instance.duration, end_time)
        
        # Generate dataset
        dataset = generate_ml_dataset(player_instance, (start_time, end_time))
        
        if dataset is None:
            return
            
        # Create and show dataset window
        from .ml_dataset_window import MLDatasetWindow
        window_title = f"ML Dataset ({start_time:.2f}s - {end_time:.2f}s)"
        dataset_window = MLDatasetWindow(window_title, dataset, player_instance)
        dataset_window.show()
        
        # Return window reference to prevent garbage collection
        return dataset_window
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Show error dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f"Error showing ML dataset window: {str(e)}")
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        return None


def show_ml_model_window(parent, dataset):
    """
    Show the ML model training window for the given dataset.
    
    Parameters
    ----------
    parent : QWidget
        Parent widget
    dataset : pandas.DataFrame
        Dataset to use for modeling
        
    Returns
    -------
    MLModelWindow or None
        The created window or None if creation failed
    """
    import time
    import traceback
    import sys
    
    def debug_log(message):
        """Print a debug message with timestamp"""
        timestamp = time.strftime('%H:%M:%S.%f')[:-3]
        print(f"[ML Window Launcher - {timestamp}] {message}", flush=True)
    
    debug_log(f"Starting ML model window creation with dataset shape {dataset.shape}")
    
    # First, add data validation and checks 
    try:
        # Basic data validation
        print(f"Dataset prepared: {len(dataset)} samples")
        
        debug_log(f"Dataset shape: {dataset.shape}")
        debug_log(f"Dataset columns: {list(dataset.columns)}")
        
        # Check if dataset is empty
        if dataset is None or len(dataset) < 10:
            debug_log("ERROR: Dataset is empty or too small")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                parent,
                "Empty Dataset",
                "The dataset is empty or has too few samples for modeling."
            )
            return None
            
        # Check if dataset has moment columns
        moment_cols = [col for col in dataset.columns if 'moment' in col.lower()]
        debug_log(f"Found {len(moment_cols)} moment columns: {moment_cols}")
        
        if not moment_cols:
            debug_log("ERROR: No moment columns found")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                parent,
                "No Target Variables",
                "No joint moment columns found in dataset.\nCannot train a model without target variables."
            )
            return None
            
        # Direct window creation with better error handling
        debug_log("Creating ML Model Window directly with strong protection")
        try:
            # Import here to avoid circular imports
            debug_log("Importing ML model window classes")
            sys.stdout.flush()
            
            # Disable parent button
            if hasattr(parent, 'train_ml_button'):
                parent.train_ml_button.setEnabled(False)
                parent.train_ml_button.setText("Loading...")
            
            # Create window directly
            from .ml_model_window import MLModelWindow
            debug_log("Creating MLModelWindow")
            window = MLModelWindow(dataset, parent)
            
            # Process events to keep UI responsive
            from PyQt5.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            
            # Show the window
            debug_log("Showing window")
            window.show()
            
            # Reset parent button
            if hasattr(parent, 'train_ml_button'):
                parent.train_ml_button.setEnabled(True)
                parent.train_ml_button.setText("Train ML Model")
            
            debug_log("Window creation successful")
            return window
            
        except Exception as e:
            debug_log(f"ERROR creating ML window: {str(e)}")
            traceback.print_exc()
            sys.stdout.flush()
            
            # Reset parent button
            if hasattr(parent, 'train_ml_button'):
                parent.train_ml_button.setEnabled(True)
                parent.train_ml_button.setText("Train ML Model")
            
            # Show error dialog
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error Creating ML Model Window")
            msg.setInformativeText(str(e))
            msg.setWindowTitle("Error")
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            
            return None
            
    except Exception as e:
        debug_log(f"ERROR in ML window launcher: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
        
        # Show error dialog
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error Launching ML Model Window")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.setDetailedText(traceback.format_exc())
        msg.exec_()
        
        return None 