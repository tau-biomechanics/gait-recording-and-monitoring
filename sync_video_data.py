#!/usr/bin/env python3
"""
Synchronized Video and Biomechanical Data Visualization

This script displays video together with biomechanical data (insole force, QTM force,
OpenCap and QTM joint angles) in synchronized plots with a cursor.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation
import matplotlib.lines as lines
import os
import glob
from scipy import signal
import opensim as osim

# Paths to data files
VIDEO_PATH = "/Volumes/T7Shield/TAU/biomechanical/sync-recording/data/opencap_data/OpenCapData_eec9c938-f468-478d-aab1-4ffcb0963207_1507-3/Videos/Cam1/InputMedia/1507-3/1507-3_sync.mp4"
DATA_DIR = "/Volumes/T7Shield/TAU/biomechanical/sync-recording/data"
OPENCAP_FILE = os.path.join(
    DATA_DIR,
    "opencap_data/OpenCapData_eec9c938-f468-478d-aab1-4ffcb0963207_1507-3/OpenSimData/Kinematics/1507-3.mot",
)
QTM_FILE = os.path.join(DATA_DIR, "qtm_data/qtm_data_20250324_182608.csv")
INSOLE_FILE = os.path.join(DATA_DIR, "insole_data/insole_data_20250324_182608.csv")


# Function to load different data sources
def load_opencap_data(file_path):
    """Load OpenCap joint angle data from .mot file"""
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

        print(
            f"Loaded OpenCap data with {len(data)} rows and {len(data.columns)} columns"
        )
        return data
    except Exception as e:
        print(f"Error loading OpenCap data: {str(e)}")
        return None


def load_qtm_data(file_path):
    """Load QTM data from CSV file"""
    print(f"Loading QTM data from: {file_path}")
    try:
        qtm_data = pd.read_csv(file_path)
        print(
            f"Loaded QTM data with {len(qtm_data)} rows and {len(qtm_data.columns)} columns"
        )

        # Convert timestamp to seconds
        if "Timestamp" in qtm_data.columns:
            qtm_data["TimeSeconds"] = qtm_data["Timestamp"] / 1000000

        return qtm_data
    except Exception as e:
        print(f"Error loading QTM data: {str(e)}")
        return None


def load_insole_data(file_path):
    """Load insole data from CSV file"""
    print(f"Loading insole data from: {file_path}")
    try:
        with open(file_path, "r") as f:
            import csv

            reader = csv.reader(f)
            headers = next(reader)  # Read headers

            # Extract only the data we need
            timestamps = []
            left_forces = []  # Will store data from column Z (index 25)
            right_forces = [] # Will store data from column AY (index 50)
            
            # Define target column indices
            timestamp_col_idx = 0
            left_force_col_idx = 25 # Excel column Z
            right_force_col_idx = 50 # Excel column AY

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
                    "Left_Force": left_forces, # Updated name
                    "Right_Force": right_forces, # Updated name
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


def extract_qtm_joint_angles(qtm_data):
    """Extract knee joint angles from QTM data"""
    # Try to find knee angle columns in QTM data
    knee_cols = [col for col in qtm_data.columns if "knee" in col.lower()]

    if not knee_cols:
        print("No knee angle columns found in QTM data")
        # Calculate knee angles using marker positions
        knee_angle_data = calculate_knee_angles_from_markers(qtm_data)
        return knee_angle_data
    else:
        print(f"Found knee angle columns in QTM data: {knee_cols}")
        # Return subset with time and knee angles
        return qtm_data[["TimeSeconds"] + knee_cols]


def calculate_knee_angles_from_markers(qtm_data):
    """Calculate knee angles from marker positions in QTM data"""
    print("Calculating knee angles from QTM marker positions")

    # Identify thigh, knee and ankle markers
    knee_angle_data = pd.DataFrame()
    knee_angle_data["TimeSeconds"] = qtm_data["TimeSeconds"]

    # Map QTM markers to anatomical points
    # Looking for thigh, knee, and ankle markers
    right_markers = {
        "thigh": [col for col in qtm_data.columns if "Marker13" in col and "_X" in col],
        "knee": [col for col in qtm_data.columns if "Marker14" in col and "_X" in col],
        "ankle": [col for col in qtm_data.columns if "Marker17" in col and "_X" in col],
    }

    left_markers = {
        "thigh": [col for col in qtm_data.columns if "Marker5" in col and "_X" in col],
        "knee": [col for col in qtm_data.columns if "Marker6" in col and "_X" in col],
        "ankle": [col for col in qtm_data.columns if "Marker9" in col and "_X" in col],
    }

    # Calculate right knee angle if markers are available
    if all(len(right_markers[k]) > 0 for k in ["thigh", "knee", "ankle"]):
        print("Calculating right knee angle from markers")
        # Extract marker positions for right leg
        thigh_marker = right_markers["thigh"][0].replace("_X", "")
        knee_marker = right_markers["knee"][0].replace("_X", "")
        ankle_marker = right_markers["ankle"][0].replace("_X", "")

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

        # Add to dataframe
        knee_angle_data["knee_angle_r"] = angles

    # Calculate left knee angle if markers are available
    if all(len(left_markers[k]) > 0 for k in ["thigh", "knee", "ankle"]):
        print("Calculating left knee angle from markers")
        # Extract marker positions for left leg
        thigh_marker = left_markers["thigh"][0].replace("_X", "")
        knee_marker = left_markers["knee"][0].replace("_X", "")
        ankle_marker = left_markers["ankle"][0].replace("_X", "")

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

        # Add to dataframe
        knee_angle_data["knee_angle_l"] = angles

    return knee_angle_data


def extract_qtm_force_data(qtm_data):
    """Extract force data from QTM data"""
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

    # Choose the force column to use (may need adjustment based on data)
    force_col = force_cols[0]
    print(f"Using QTM force column: {force_col}")

    # Calculate resultant force if X, Y, Z components available
    if (
        "ForcePlate1_ForceX" in qtm_data.columns
        and "ForcePlate1_ForceY" in qtm_data.columns
        and "ForcePlate1_ForceZ" in qtm_data.columns
    ):
        print("Calculating resultant force from X, Y, Z components")
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
    else:
        # Return time and the selected force column
        return pd.DataFrame(
            {
                "TimeSeconds": qtm_data["TimeSeconds"],
                "Force": pd.to_numeric(qtm_data[force_col], errors="coerce")
                .fillna(0)
                .values,
            }
        )


def extract_opencap_knee_angles(opencap_data):
    """Extract knee angles from OpenCap data"""
    # Find knee angle columns
    knee_cols = [col for col in opencap_data.columns if "knee" in col.lower()]

    if not knee_cols:
        print("No knee angle columns found in OpenCap data")
        return None

    print(f"Found knee angle columns in OpenCap data: {knee_cols}")

    # Return subset with time and knee angles
    return opencap_data[["time"] + knee_cols]


def synchronize_data(video_fps, *data_frames):
    """
    Synchronize multiple data frames to a common time base

    Parameters:
    -----------
    video_fps : float
        Frames per second of the video
    *data_frames : list of (DataFrame, str)
        List of tuples containing (dataframe, time_column_name)

    Returns:
    --------
    list of DataFrames with synchronized time values
    """
    # Find the common time range across all data frames
    min_times = []
    max_times = []

    for df, time_col in data_frames:
        if df is not None and not df.empty:
            min_times.append(df[time_col].min())
            max_times.append(df[time_col].max())

    if not min_times or not max_times:
        print("Error: No valid data frames to synchronize")
        return None

    common_min_time = max(min_times)
    common_max_time = min(max_times)

    print(f"Common time range: {common_min_time:.2f}s to {common_max_time:.2f}s")

    # Create a common time base for all data
    # Match to video frames for perfect synchronization
    video_duration = common_max_time - common_min_time
    num_frames = int(video_duration * video_fps)
    common_time = np.linspace(common_min_time, common_max_time, num_frames)

    # Resample/interpolate each data frame to the common time base
    synchronized_frames = []

    for df, time_col in data_frames:
        if df is not None and not df.empty:
            # Create new DataFrame with synchronized time
            sync_df = pd.DataFrame({"time": common_time})

            # Interpolate all columns to match the common time
            for col in df.columns:
                if col != time_col:
                    # Use linear interpolation to map values to new time points
                    sync_df[col] = np.interp(
                        common_time,
                        df[time_col].values,
                        df[col].values,
                        left=np.nan,
                        right=np.nan,
                    )

            synchronized_frames.append(sync_df)
        else:
            synchronized_frames.append(None)

    return synchronized_frames, common_time


def filter_signal(data, fs=100, cutoff=10, order=4):
    """Apply a low-pass Butterworth filter to smooth signals"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)

    # Handle NaN values
    valid_mask = ~np.isnan(data)
    filtered_data = np.copy(data)

    if np.sum(valid_mask) > 0:  # Only filter if there's valid data
        filtered_data[valid_mask] = signal.filtfilt(b, a, data[valid_mask])

    return filtered_data


class SynchronizedVideoPlayer:
    def __init__(
        self, video_path, insole_data, qtm_force_data, opencap_knee_data, qtm_knee_data
    ):
        self.video_path = video_path
        self.insole_data = insole_data
        self.qtm_force_data = qtm_force_data
        self.opencap_knee_data = opencap_knee_data
        self.qtm_knee_data = qtm_knee_data

        # Initialize time offsets for each data stream (in seconds)
        self.insole_offset = 3.10
        self.qtm_force_offset = 3.27
        self.opencap_knee_offset = 0.0
        self.qtm_knee_offset = 3.27

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.current_frame = 0

        print(
            f"Video properties: {self.frame_count} frames, {self.fps} fps, {self.duration:.2f}s duration"
        )

        # Synchronize data
        data_frames = []
        if self.insole_data is not None:
            data_frames.append((self.insole_data, "TimeSeconds"))
        if self.qtm_force_data is not None:
            data_frames.append((self.qtm_force_data, "TimeSeconds"))
        if self.opencap_knee_data is not None:
            data_frames.append((self.opencap_knee_data, "time"))
        if self.qtm_knee_data is not None:
            data_frames.append((self.qtm_knee_data, "TimeSeconds"))

        self.sync_frames, self.common_time = synchronize_data(self.fps, *data_frames)

        # Set up the figure and subplots with more space for controls
        self.fig = plt.figure(figsize=(18, 10))

        # Create a grid with 3 columns - video, plots, offset controls
        self.gs = GridSpec(1, 3, width_ratios=[1, 1.2, 0.3], figure=self.fig)

        # Video frame on the left
        self.ax_video = self.fig.add_subplot(self.gs[0, 0])
        self.ax_video.set_axis_off()

        # Create a sub-grid for the charts in the middle
        self.gs_middle = GridSpec(
            6,
            1,
            height_ratios=[1, 1, 1, 1, 0.3, 0.2],
            figure=self.fig,
            left=self.gs[0, 1].get_position(self.fig).x0,
            right=self.gs[0, 1].get_position(self.fig).x1,
            bottom=self.gs[0, 1].get_position(self.fig).y0,
            top=self.gs[0, 1].get_position(self.fig).y1,
        )

        # Create a sub-grid for the offset controls on the right
        self.gs_right = GridSpec(
            5,
            1,
            height_ratios=[1, 1, 1, 1, 0.3],
            figure=self.fig,
            left=self.gs[0, 2].get_position(self.fig).x0,
            right=self.gs[0, 2].get_position(self.fig).x1,
            bottom=self.gs[0, 2].get_position(self.fig).y0,
            top=self.gs[0, 2].get_position(self.fig).y1,
        )

        # Initialize with the first frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get video dimensions to keep aspect ratio
            height, width = frame.shape[:2]
            aspect_ratio = width / height

            # Display with original aspect ratio
            self.video_image = self.ax_video.imshow(frame, aspect="equal")

            # Set axis limits to maintain aspect ratio
            self.ax_video.set_xlim([-width * 0.05, width * 1.05])
            self.ax_video.set_ylim([height * 1.05, -height * 0.05])

        # Initialize data plots in the middle
        self.ax_insole = self.fig.add_subplot(self.gs_middle[0, 0])
        self.ax_qtm_force = self.fig.add_subplot(
            self.gs_middle[1, 0], sharex=self.ax_insole
        )
        self.ax_opencap_knee = self.fig.add_subplot(
            self.gs_middle[2, 0], sharex=self.ax_insole
        )
        self.ax_qtm_knee = self.fig.add_subplot(
            self.gs_middle[3, 0], sharex=self.ax_insole
        )

        # Set up time slider
        self.ax_slider = self.fig.add_subplot(self.gs_middle[4, 0])
        self.slider = Slider(
            self.ax_slider, "Time (s)", 0, self.duration, valinit=0, valfmt="%1.2f"
        )
        self.slider.on_changed(self.on_slider_change)

        # Set up time window controls below the slider
        self.ax_time_window = self.fig.add_subplot(self.gs_middle[5, 0])
        self.ax_time_window.set_axis_off()  # Hide axes for this container
        # Add TextBoxes for start and end time
        ax_start_text = self.fig.add_axes(
            [0.38, 0.02, 0.08, 0.04]
        )  # Position: [left, bottom, width, height]
        ax_end_text = self.fig.add_axes([0.50, 0.02, 0.08, 0.04])
        self.start_time_textbox = TextBox(ax_start_text, "Start:", initial=f"0.00")
        self.end_time_textbox = TextBox(
            ax_end_text, "End:", initial=f"{self.duration:.2f}"
        )
        self.start_time_textbox.on_submit(self.on_time_window_change)
        self.end_time_textbox.on_submit(self.on_time_window_change)
        self.time_window_start = 0.0
        self.time_window_end = self.duration

        # Initialize offset control areas
        self.ax_insole_offset = self.fig.add_subplot(self.gs_right[0, 0])
        self.ax_qtm_force_offset = self.fig.add_subplot(self.gs_right[1, 0])
        self.ax_opencap_knee_offset = self.fig.add_subplot(self.gs_right[2, 0])
        self.ax_qtm_knee_offset = self.fig.add_subplot(self.gs_right[3, 0])

        # Initialize cursor line in each plot
        self.cursor_lines = []

        # Set up offset controls
        self.setup_offset_controls()

        # Set up the data plots
        self.setup_plots()

        # Animation
        self.is_playing = False
        self.anim = None

        # Set up keyboard listeners
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def setup_offset_controls(self):
        """Set up sliders and text fields for data offsets"""
        # Maximum offset range in seconds (±2 seconds)
        offset_range = 2.0

        # Insole data offset controls
        self.ax_insole_offset.set_title("Insole Offset")
        self.insole_offset_slider = Slider(
            self.ax_insole_offset,
            "sec",
            -offset_range,
            offset_range,
            valinit=self.insole_offset,
            valfmt="%1.2f",
        )
        self.insole_offset_slider.on_changed(self.on_insole_offset_change)

        # Add text box for insole offset under the slider
        text_left = self.ax_insole_offset.get_position().x0
        text_bottom = self.ax_insole_offset.get_position().y0 - 0.03
        text_width = 0.1
        text_height = 0.03
        self.ax_insole_text = plt.axes(
            [text_left, text_bottom, text_width, text_height]
        )
        self.insole_offset_textbox = TextBox(
            self.ax_insole_text, "Value: ", initial=f"{self.insole_offset:.2f}"
        )
        self.insole_offset_textbox.on_submit(self.on_insole_offset_text_change)

        # QTM force data offset controls
        self.ax_qtm_force_offset.set_title("QTM Force Offset")
        self.qtm_force_offset_slider = Slider(
            self.ax_qtm_force_offset,
            "sec",
            -offset_range,
            offset_range,
            valinit=self.qtm_force_offset,
            valfmt="%1.2f",
        )
        self.qtm_force_offset_slider.on_changed(self.on_qtm_force_offset_change)

        # Add text box for QTM force offset
        text_bottom = self.ax_qtm_force_offset.get_position().y0 - 0.03
        self.ax_qtm_force_text = plt.axes(
            [text_left, text_bottom, text_width, text_height]
        )
        self.qtm_force_offset_textbox = TextBox(
            self.ax_qtm_force_text, "Value: ", initial=f"{self.qtm_force_offset:.2f}"
        )
        self.qtm_force_offset_textbox.on_submit(self.on_qtm_force_offset_text_change)

        # OpenCap knee data offset controls
        self.ax_opencap_knee_offset.set_title("OpenCap Offset")
        self.opencap_knee_offset_slider = Slider(
            self.ax_opencap_knee_offset,
            "sec",
            -offset_range,
            offset_range,
            valinit=self.opencap_knee_offset,
            valfmt="%1.2f",
        )
        self.opencap_knee_offset_slider.on_changed(self.on_opencap_knee_offset_change)

        # Add text box for OpenCap knee offset
        text_bottom = self.ax_opencap_knee_offset.get_position().y0 - 0.03
        self.ax_opencap_knee_text = plt.axes(
            [text_left, text_bottom, text_width, text_height]
        )
        self.opencap_knee_offset_textbox = TextBox(
            self.ax_opencap_knee_text,
            "Value: ",
            initial=f"{self.opencap_knee_offset:.2f}",
        )
        self.opencap_knee_offset_textbox.on_submit(
            self.on_opencap_knee_offset_text_change
        )

        # QTM knee data offset controls
        self.ax_qtm_knee_offset.set_title("QTM Knee Offset")
        self.qtm_knee_offset_slider = Slider(
            self.ax_qtm_knee_offset,
            "sec",
            -offset_range,
            offset_range,
            valinit=self.qtm_knee_offset,
            valfmt="%1.2f",
        )
        self.qtm_knee_offset_slider.on_changed(self.on_qtm_knee_offset_change)

        # Add text box for QTM knee offset
        text_bottom = self.ax_qtm_knee_offset.get_position().y0 - 0.03
        self.ax_qtm_knee_text = plt.axes(
            [text_left, text_bottom, text_width, text_height]
        )
        self.qtm_knee_offset_textbox = TextBox(
            self.ax_qtm_knee_text, "Value: ", initial=f"{self.qtm_knee_offset:.2f}"
        )
        self.qtm_knee_offset_textbox.on_submit(self.on_qtm_knee_offset_text_change)

    def on_insole_offset_change(self, val):
        """Handle insole offset slider change"""
        self.insole_offset = val
        self.insole_offset_textbox.set_val(f"{val:.2f}")
        self.update_plots()

    def on_insole_offset_text_change(self, val):
        """Handle insole offset text input change"""
        try:
            offset = float(val)
            self.insole_offset = offset
            self.insole_offset_slider.set_val(offset)
            self.update_plots()
        except ValueError:
            # Restore original value if input is invalid
            self.insole_offset_textbox.set_val(f"{self.insole_offset:.2f}")

    def on_qtm_force_offset_change(self, val):
        """Handle QTM force offset slider change"""
        self.qtm_force_offset = val
        self.qtm_force_offset_textbox.set_val(f"{val:.2f}")
        self.update_plots()

    def on_qtm_force_offset_text_change(self, val):
        """Handle QTM force offset text input change"""
        try:
            offset = float(val)
            self.qtm_force_offset = offset
            self.qtm_force_offset_slider.set_val(offset)
            self.update_plots()
        except ValueError:
            # Restore original value if input is invalid
            self.qtm_force_offset_textbox.set_val(f"{self.qtm_force_offset:.2f}")

    def on_opencap_knee_offset_change(self, val):
        """Handle OpenCap knee offset slider change"""
        self.opencap_knee_offset = val
        self.opencap_knee_offset_textbox.set_val(f"{val:.2f}")
        self.update_plots()

    def on_opencap_knee_offset_text_change(self, val):
        """Handle OpenCap knee offset text input change"""
        try:
            offset = float(val)
            self.opencap_knee_offset = offset
            self.opencap_knee_offset_slider.set_val(offset)
            self.update_plots()
        except ValueError:
            # Restore original value if input is invalid
            self.opencap_knee_offset_textbox.set_val(f"{self.opencap_knee_offset:.2f}")

    def on_qtm_knee_offset_change(self, val):
        """Handle QTM knee offset slider change"""
        self.qtm_knee_offset = val
        self.qtm_knee_offset_textbox.set_val(f"{val:.2f}")
        self.update_plots()

    def on_qtm_knee_offset_text_change(self, val):
        """Handle QTM knee offset text input change"""
        try:
            offset = float(val)
            self.qtm_knee_offset = offset
            self.qtm_knee_offset_slider.set_val(offset)
            self.update_plots()
        except ValueError:
            # Restore original value if input is invalid
            self.qtm_knee_offset_textbox.set_val(f"{self.qtm_knee_offset:.2f}")

    def on_time_window_change(self, val):
        """Handle changes to the start or end time text boxes"""
        try:
            start_time = float(self.start_time_textbox.text)
            end_time = float(self.end_time_textbox.text)

            # Ensure start time is less than end time and within bounds
            start_time = max(0.0, start_time)
            end_time = min(self.duration, end_time)
            if start_time >= end_time:
                end_time = start_time + 0.1  # Ensure minimum duration
                end_time = min(self.duration, end_time)

            self.time_window_start = start_time
            self.time_window_end = end_time

            # Update text boxes with validated values
            self.start_time_textbox.set_val(f"{self.time_window_start:.2f}")
            self.end_time_textbox.set_val(f"{self.time_window_end:.2f}")

            # Update the plots to reflect the new time window
            self.update_plots()

        except ValueError:
            # Restore previous valid values if input is invalid
            self.start_time_textbox.set_val(f"{self.time_window_start:.2f}")
            self.end_time_textbox.set_val(f"{self.time_window_end:.2f}")

    def update_plots(self):
        """Update all plots with current offsets and time window"""
        # Clear existing plots
        self.ax_insole.clear()
        self.ax_qtm_force.clear()
        self.ax_opencap_knee.clear()
        self.ax_qtm_knee.clear()

        # Rebuild the plots with offsets and apply time window
        self.setup_plots()

        # Restore cursor position
        current_time = self.current_frame / self.fps
        for line in self.cursor_lines:
            line.set_xdata([current_time, current_time])

        # Redraw the figure
        self.fig.canvas.draw_idle()

    def setup_plots(self):
        """Set up the data plots with the synchronized data and time offsets"""
        # Reset cursor lines list
        self.cursor_lines = []

        idx = 0

        # Insole force plot
        if self.insole_data is not None:
            insole_sync = self.sync_frames[idx]
            idx += 1

            # Apply filtering to smooth the signals
            left_force = filter_signal(insole_sync["Left_Force"].values)
            right_force = filter_signal(insole_sync["Right_Force"].values)

            # Apply offset to time values
            adjusted_time = insole_sync["time"] + self.insole_offset

            (self.insole_left_line,) = self.ax_insole.plot(
                adjusted_time, left_force, "g-", label="Left Force", linewidth=0.8
            )
            (self.insole_right_line,) = self.ax_insole.plot(
                adjusted_time, right_force, "r-", label="Right Force", linewidth=0.8
            )

            self.ax_insole.set_ylabel("Insole Force (N)")
            self.ax_insole.legend(loc="upper right")
            self.ax_insole.grid(True)

            # Add cursor line
            cursor_line = self.ax_insole.axvline(
                x=0, color="k", linestyle="--", linewidth=0.8
            )
            self.cursor_lines.append(cursor_line)

        # QTM force plot
        if self.qtm_force_data is not None:
            qtm_force_sync = self.sync_frames[idx]
            idx += 1

            # Apply filtering
            force = filter_signal(qtm_force_sync["Force"].values)

            # Apply offset to time values
            adjusted_time = qtm_force_sync["time"] + self.qtm_force_offset

            (self.qtm_force_line,) = self.ax_qtm_force.plot(
                adjusted_time, force, "b-", label="Force", linewidth=0.8
            )

            self.ax_qtm_force.set_ylabel("QTM Force (N)")
            self.ax_qtm_force.legend(loc="upper right")
            self.ax_qtm_force.grid(True)

            # Add cursor line
            cursor_line = self.ax_qtm_force.axvline(
                x=0, color="k", linestyle="--", linewidth=0.8
            )
            self.cursor_lines.append(cursor_line)

        # OpenCap knee angles plot
        if self.opencap_knee_data is not None:
            opencap_knee_sync = self.sync_frames[idx]
            idx += 1

            # Apply offset to time values
            adjusted_time = opencap_knee_sync["time"] + self.opencap_knee_offset

            # Plot each knee angle column
            knee_cols = [
                col for col in opencap_knee_sync.columns if "knee" in col.lower()
            ]
            for col in knee_cols:
                # Apply filtering
                angles = filter_signal(opencap_knee_sync[col].values)

                label = col.replace("_", " ").title()
                self.ax_opencap_knee.plot(
                    adjusted_time, angles, label=label, linewidth=0.8
                )

            self.ax_opencap_knee.set_ylabel("OpenCap Knee Angle (°)")
            self.ax_opencap_knee.legend(loc="upper right")
            self.ax_opencap_knee.grid(True)

            # Add cursor line
            cursor_line = self.ax_opencap_knee.axvline(
                x=0, color="k", linestyle="--", linewidth=0.8
            )
            self.cursor_lines.append(cursor_line)

        # QTM knee angles plot
        if self.qtm_knee_data is not None:
            qtm_knee_sync = self.sync_frames[idx]
            idx += 1

            # Apply offset to time values
            adjusted_time = qtm_knee_sync["time"] + self.qtm_knee_offset

            # Plot each knee angle column
            knee_cols = [col for col in qtm_knee_sync.columns if "knee" in col.lower()]
            for col in knee_cols:
                # Apply filtering
                angles = filter_signal(qtm_knee_sync[col].values)
                angles = -angles + 180

                label = col.replace("_", " ").title()
                self.ax_qtm_knee.plot(adjusted_time, angles, label=label, linewidth=0.8)

            self.ax_qtm_knee.set_ylabel("QTM Knee Angle (°)")
            self.ax_qtm_knee.legend(loc="upper right")
            self.ax_qtm_knee.grid(True)

            # Add cursor line
            cursor_line = self.ax_qtm_knee.axvline(
                x=0, color="k", linestyle="--", linewidth=0.8
            )
            self.cursor_lines.append(cursor_line)

        # Set common x-axis properties
        self.ax_qtm_knee.set_xlabel("Time (s)")

        # Apply time window limits to all plot axes
        for ax in [
            self.ax_insole,
            self.ax_qtm_force,
            self.ax_opencap_knee,
            self.ax_qtm_knee,
        ]:
            ax.set_xlim(self.time_window_start, self.time_window_end)

        # Apply tight layout
        plt.tight_layout()

    def update_frame(self, frame=None):
        """Update the display with the specified frame"""
        if frame is not None:
            self.current_frame = frame

        # Set video to the current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if ret:
            # Update video frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_image.set_array(frame_rgb)

            # Calculate current time
            current_time = self.current_frame / self.fps

            # Update cursor position in all plots
            for line in self.cursor_lines:
                line.set_xdata([current_time, current_time])  # Pass as a sequence

            # Update slider value without triggering callback
            self.slider.set_val(current_time)

            # Update title with current time
            self.ax_video.set_title(
                f"Frame: {self.current_frame}/{self.frame_count}, Time: {current_time:.2f}s"
            )

        return [self.video_image] + self.cursor_lines

    def on_slider_change(self, val):
        """Handle slider value change"""
        frame = int(val * self.fps)
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_frame()
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == " ":  # Space bar
            # Toggle play/pause
            if self.is_playing:
                self.pause()
            else:
                self.play()
        elif event.key == "right":
            # Advance one frame
            self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
            self.update_frame()
            self.fig.canvas.draw_idle()
        elif event.key == "left":
            # Go back one frame
            self.current_frame = max(self.current_frame - 1, 0)
            self.update_frame()
            self.fig.canvas.draw_idle()

    def play(self):
        """Start playing the video"""
        if not self.is_playing:
            self.is_playing = True

            def animate(frame_idx):
                # Calculate frame to display
                actual_frame = self.current_frame + frame_idx
                if actual_frame >= self.frame_count:
                    self.pause()
                    return self.update_frame(self.current_frame)

                return self.update_frame(actual_frame)

            self.anim = animation.FuncAnimation(
                self.fig,
                animate,
                interval=1000 / self.fps,
                blit=True,
                cache_frame_data=False,
            )

    def pause(self):
        """Pause the video"""
        if self.is_playing:
            self.is_playing = False
            if self.anim is not None:
                self.anim.event_source.stop()

    def show(self):
        """Show the figure"""
        plt.show()

    def close(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()


def main():
    # Load data
    opencap_data = load_opencap_data(OPENCAP_FILE)
    qtm_data = load_qtm_data(QTM_FILE)
    insole_data = load_insole_data(INSOLE_FILE)

    # Extract specific data
    opencap_knee_data = None
    if opencap_data is not None:
        opencap_knee_data = extract_opencap_knee_angles(opencap_data)
    qtm_force_data = extract_qtm_force_data(qtm_data)
    qtm_knee_data = extract_qtm_joint_angles(qtm_data)

    # Create the synchronized video player
    player = SynchronizedVideoPlayer(
        VIDEO_PATH, insole_data, qtm_force_data, opencap_knee_data, qtm_knee_data
    )

    # Show the player
    print("\nVideo Player Controls:")
    print("- Space: Play/Pause")
    print("- Left/Right Arrow: Previous/Next Frame")
    print("- Slider: Seek through video")
    player.show()

    # Clean up
    player.close()


if __name__ == "__main__":
    main()
