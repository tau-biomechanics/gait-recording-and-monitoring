"""
Module for handling correlation analysis in the synchronized video player
"""

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from ..utils.signal_processing import filter_signal
from ..processors.correlation_analysis import calculate_similarity_metrics
from .correlation_window import CorrelationPlotWindow


def analyze_correlation(parent, selected_data, time_range, subject_mass=70.0):
    """
    Analyze correlation between two data series and display results
    
    Parameters:
    -----------
    parent : QWidget
        Parent widget for the correlation window
    selected_data : list
        List of tuples containing (name, time, values) for selected data series
    time_range : tuple
        Tuple containing (start_time, end_time) for the analysis
    subject_mass : float, optional
        Subject mass for normalization (default: 70.0 kg)
    
    Returns:
    --------
    bool
        True if analysis was successful, False otherwise
    """
    try:
        start_time, end_time = time_range
        
        # Count how many data series are selected
        selected_count = len(selected_data)
        
        # Check if exactly 2 data series are selected
        if selected_count != 2:
            # Show error message in a dialog
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select exactly 2 data series for correlation analysis.")
            msg.setWindowTitle("Selection Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        
        # Extract the data
        series1_name, time1, values1 = selected_data[0]
        series2_name, time2, values2 = selected_data[1]
        
        print(f"Calculating correlation between {series1_name} and {series2_name}")
        
        # Trim data to the selected time window
        mask1 = (time1 >= start_time) & (time1 <= end_time)
        mask2 = (time2 >= start_time) & (time2 <= end_time)
        
        time1_window = time1[mask1]
        values1_window = values1[mask1]
        time2_window = time2[mask2]
        values2_window = values2[mask2]
        
        # Check if we have enough data
        if len(time1_window) < 2 or len(time2_window) < 2:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Not enough data points in the selected time window.")
            msg.setWindowTitle("Data Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        
        # Calculate correlation metrics
        metrics = calculate_similarity_metrics(
            values1_window, time1_window, 
            values2_window, time2_window
        )
        
        # Create and show the correlation window
        window_title = f"Correlation Analysis: {series1_name} vs {series2_name} ({start_time:.2f}s - {end_time:.2f}s)"
        correlation_window = CorrelationPlotWindow(
            window_title, metrics, series1_name, series2_name, parent
        )
        correlation_window.show()
        return True
    
    except Exception as e:
        # Show error dialog
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f"Error in correlation analysis: {str(e)}")
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        import traceback
        traceback.print_exc()
        return False


def collect_selected_data(player):
    """
    Collect data from selected checkboxes in the player UI
    
    Parameters:
    -----------
    player : QtSynchronizedVideoPlayer
        The player instance with selection UI
    
    Returns:
    --------
    list
        List of tuples containing (name, time, values) for selected series
    """
    selected_data = []
    
    # Insole data
    if player.insole_data is not None:
        insole_sync = player.sync_frames[0] if len(player.sync_frames) > 0 else None
        if insole_sync is not None:
            if hasattr(player, 'insole_left_check') and player.insole_left_check.isChecked():
                adjusted_time = insole_sync["time"] + player.insole_offset
                left_force = filter_signal(insole_sync["Left_Force"].values)
                selected_data.append(("Insole Left Force", adjusted_time, left_force))
            
            if hasattr(player, 'insole_right_check') and player.insole_right_check.isChecked():
                adjusted_time = insole_sync["time"] + player.insole_offset
                right_force = filter_signal(insole_sync["Right_Force"].values)
                selected_data.append(("Insole Right Force", adjusted_time, right_force))
    
    # QTM force data
    if player.qtm_force_data is not None:
        idx = 1 if player.insole_data is not None else 0
        if idx < len(player.sync_frames):
            qtm_force_sync = player.sync_frames[idx]
            if hasattr(player, 'qtm_force_check') and player.qtm_force_check.isChecked():
                adjusted_time = qtm_force_sync["time"] + player.qtm_force_offset
                force = filter_signal(qtm_force_sync["Force"].values)
                selected_data.append(("QTM Force", adjusted_time, force))
    
    # OpenCap joint data
    if player.opencap_joint_data is not None:
        idx = (1 if player.insole_data is not None else 0) + (1 if player.qtm_force_data is not None else 0)
        if idx < len(player.sync_frames):
            opencap_joint_sync = player.sync_frames[idx]
            if hasattr(player, 'opencap_checks'):
                for param, check in player.opencap_checks.items():
                    if check.isChecked():
                        adjusted_time = opencap_joint_sync["time"] + player.opencap_joint_offset
                        angles = filter_signal(opencap_joint_sync[param].values)
                        label = "OpenCap " + param.replace("_", " ").title()
                        selected_data.append((label, adjusted_time, angles))
    
    # QTM joint data
    if player.qtm_joint_data is not None:
        idx = (1 if player.insole_data is not None else 0) + \
              (1 if player.qtm_force_data is not None else 0) + \
              (1 if player.opencap_joint_data is not None else 0)
        if idx < len(player.sync_frames):
            qtm_joint_sync = player.sync_frames[idx]
            if hasattr(player, 'qtm_checks'):
                for param, check in player.qtm_checks.items():
                    if check.isChecked():
                        adjusted_time = qtm_joint_sync["time"] + player.qtm_joint_offset
                        angles = filter_signal(qtm_joint_sync[param].values)
                        
                        # Only apply the 180 degree adjustment for knee angles
                        if "knee" in param.lower():
                            angles = -angles + 180  # Invert direction to match OpenCap convention
                        
                        label = "QTM " + param.replace("_", " ").title()
                        selected_data.append((label, adjusted_time, angles))
    
    # Joint Moment data
    if player.qtm_force_data is not None and player.qtm_joint_data is not None:
        if hasattr(player, 'joint_moment_checks'):
            for param, check in player.joint_moment_checks.items():
                if check.isChecked():
                    # Find indices for force and joint data
                    force_idx = 1 if player.insole_data is not None else 0
                    joint_idx = (1 if player.insole_data is not None else 0) + \
                               (1 if player.qtm_force_data is not None else 0) + \
                               (1 if player.opencap_joint_data is not None else 0)
                    
                    if force_idx < len(player.sync_frames) and joint_idx < len(player.sync_frames):
                        qtm_force_sync = player.sync_frames[force_idx]
                        qtm_joint_sync = player.sync_frames[joint_idx]
                        
                        # Get adjusted time
                        adjusted_time = qtm_force_sync["time"] + player.qtm_force_offset
                        
                        # Get the associated joint angle parameter
                        angle_param = None
                        if "Ankle" in param and "Ankle" in player.selected_qtm_param:
                            angle_param = player.selected_qtm_param
                        elif "Knee" in param and "Knee" in player.selected_qtm_param:
                            angle_param = player.selected_qtm_param
                        elif "Hip" in param and "Hip" in player.selected_qtm_param:
                            angle_param = player.selected_qtm_param
                        
                        if angle_param is not None and angle_param in qtm_joint_sync.columns:
                            forces = qtm_force_sync["Force"].values
                            angles = qtm_joint_sync[angle_param].values
                            
                            # Adjust for knee angles if needed
                            if "knee" in angle_param.lower():
                                angles = -angles + 180
                            
                            # Calculate moments from the player's calculate_joint_moments function
                            from ..gui.qt_player import calculate_joint_moments
                            moments = calculate_joint_moments(
                                forces,
                                angles,
                                player.subject_mass
                            )
                            
                            label = param.replace("_", " ").title()
                            selected_data.append((label, adjusted_time, moments))
    
    return selected_data 