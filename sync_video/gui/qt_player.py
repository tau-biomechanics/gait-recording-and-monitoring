"""
Qt-based synchronized video player implementation
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.lines as lines
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QPushButton,
    QLineEdit,
    QSplitter,
    QFrame,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QSizePolicy,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence

# Add import for biomechanical calculations
from scipy import signal
import scipy.constants as constants

# Import our correlation classes
from .correlation_window import CorrelationPlotWindow
from ..processors.correlation_analysis import calculate_similarity_metrics
from .correlation_module import analyze_correlation, collect_selected_data

from ..utils.signal_processing import filter_signal
from ..utils.synchronization import synchronize_data
from ..config import (
    DEFAULT_INSOLE_OFFSET,
    DEFAULT_QTM_FORCE_OFFSET,
    DEFAULT_OPENCAP_KNEE_OFFSET,
    DEFAULT_QTM_KNEE_OFFSET,
)

# Add joint moment calculation function
def calculate_joint_moments(forces, joint_angles, subject_mass=70.0):
    """
    Calculate joint moments using simplified biomechanical principles
    
    Parameters:
    -----------
    forces : numpy.ndarray
        Ground reaction force data [N]
    joint_angles : numpy.ndarray
        Joint angle data in degrees
    subject_mass : float
        Subject mass in kg for normalization
        
    Returns:
    --------
    numpy.ndarray
        Calculated joint moments in Nm/kg
    """
    try:
        print(f"Calculating joint moments: {len(forces)} forces, {len(joint_angles)} angles")
        
        # Convert forces and angles to numpy arrays if they aren't already
        forces = np.asarray(forces)
        joint_angles = np.asarray(joint_angles)
        
        # Replace NaN values with interpolated values or zeros
        # First check if we have NaN values
        if np.isnan(joint_angles).any():
            print(f"Warning: Found {np.sum(np.isnan(joint_angles))} NaN values in joint angles")
            
            # Create an array of indices
            indices = np.arange(len(joint_angles))
            
            # Get indices of non-NaN values
            valid_indices = ~np.isnan(joint_angles)
            
            if np.sum(valid_indices) > 1:  # Need at least 2 points for interpolation
                # Interpolate NaN values if we have some valid data
                joint_angles = np.interp(
                    indices, 
                    indices[valid_indices], 
                    joint_angles[valid_indices]
                )
                print(f"Interpolated joint angles, new range: {np.min(joint_angles)}-{np.max(joint_angles)}")
            else:
                # Not enough valid data for interpolation, use zeros
                print("Not enough valid angle data for interpolation, using zeros")
                joint_angles = np.zeros_like(joint_angles)
        
        # Check for NaN in forces
        if np.isnan(forces).any():
            print(f"Warning: Found {np.sum(np.isnan(forces))} NaN values in forces")
            
            # Create an array of indices
            indices = np.arange(len(forces))
            
            # Get indices of non-NaN values
            valid_indices = ~np.isnan(forces)
            
            if np.sum(valid_indices) > 1:  # Need at least 2 points for interpolation
                # Interpolate NaN values
                forces = np.interp(
                    indices, 
                    indices[valid_indices], 
                    forces[valid_indices]
                )
                print(f"Interpolated forces, new range: {np.min(forces)}-{np.max(forces)}")
            else:
                # Not enough valid data for interpolation, use zeros
                print("Not enough valid force data for interpolation, using zeros")
                forces = np.zeros_like(forces)
        
        # Basic sanity checks
        if len(forces) < 2 or len(joint_angles) < 2:
            print("Not enough data points for moment calculation")
            return np.array([])
            
        # Convert angles to radians for calculation
        angles_rad = np.radians(joint_angles)
        
        # Determine moment arm based on typical segment lengths
        # For simplicity, use a constant lever arm length based on the joint
        # Typical proportions of height:
        # - Ankle moment arm: ~0.05 * height
        # - Knee moment arm: ~0.25 * height
        # - Hip moment arm: ~0.4 * height
        
        # Default height (m) - can be adjusted based on subject
        subject_height = 1.75
        
        # Use a simplified calculation based on which joint is being analyzed
        # The joint type is inferred from the variable name used when calling this function
        segment_length = 0.25 * subject_height  # Default to knee
        
        # Calculate effective moment arm (distance * sin(angle))
        # This is a simplified model - in reality the moment arm changes with posture
        moment_arm = segment_length * np.sin(angles_rad)
        
        # Calculate raw moments: Force * Moment arm
        # This is a simplified calculation that approximates the moment
        raw_moments = forces * moment_arm
        
        # Normalize by subject mass (Nm/kg)
        normalized_moments = raw_moments / subject_mass
        
        # Apply smoothing filter
        # Use a lower cutoff frequency for cleaner results
        b, a = signal.butter(2, 0.05)
        smoothed_moments = signal.filtfilt(b, a, normalized_moments)
        
        # Final check for valid output
        if np.all(np.isnan(smoothed_moments)):
            print("Warning: All moment values are NaN")
            return np.array([])
        elif np.any(np.isnan(smoothed_moments)):
            print(f"Warning: {np.sum(np.isnan(smoothed_moments))} NaN values in calculated moments, replacing with zeros")
            smoothed_moments = np.nan_to_num(smoothed_moments, nan=0.0)
        
        print(f"Moment calculation successful, generated {len(smoothed_moments)} points, range: {np.min(smoothed_moments)}-{np.max(smoothed_moments)}")
        return smoothed_moments
        
    except Exception as e:
        print(f"Error in moment calculation: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plot embedding in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)


class VideoFrame(QLabel):
    """Video display frame with synchronization markers"""

    def __init__(self, parent=None):
        super(VideoFrame, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.current_frame = None

    def set_frame(self, frame):
        """Set the video frame to display"""
        if frame is None:
            return

        # Store the original frame
        self.current_frame = frame

        # Display the frame
        self.update_frame_display()

    def update_frame_display(self):
        """Update the displayed frame with current size"""
        if self.current_frame is None:
            return

        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width

        # Convert to QImage and then QPixmap
        q_img = QImage(
            frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit widget while preserving aspect ratio
        pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        """Handle resize events to update the frame display"""
        super(VideoFrame, self).resizeEvent(event)
        # Update the displayed frame when the widget is resized
        self.update_frame_display()


class DataPlotWidget(QWidget):
    """Widget for displaying synchronized data plots"""

    def __init__(self, parent=None):
        super(DataPlotWidget, self).__init__(parent)

        # Main layout
        layout = QVBoxLayout(self)

        # Create matplotlib canvases for each data type
        self.insole_canvas = MatplotlibCanvas(self, width=5, height=2)
        self.qtm_force_canvas = MatplotlibCanvas(self, width=5, height=2)
        self.opencap_angle_canvas = MatplotlibCanvas(self, width=5, height=2)
        self.qtm_angle_canvas = MatplotlibCanvas(self, width=5, height=2)
        # Add new canvas for joint moments
        self.joint_moment_canvas = MatplotlibCanvas(self, width=5, height=2)

        # Add canvases to layout
        layout.addWidget(self.insole_canvas)
        layout.addWidget(self.qtm_force_canvas)
        layout.addWidget(self.opencap_angle_canvas)
        layout.addWidget(self.qtm_angle_canvas)
        layout.addWidget(self.joint_moment_canvas)

        # Initialize plots
        self.insole_lines = []
        self.qtm_force_lines = []
        self.opencap_angle_lines = []
        self.qtm_angle_lines = []
        self.joint_moment_lines = []
        self.cursor_lines = []

        # Set axis labels
        self.insole_canvas.axes.set_ylabel("Insole Force (N)")
        self.qtm_force_canvas.axes.set_ylabel("QTM Force (N)")
        self.opencap_angle_canvas.axes.set_ylabel("OpenCap Angle (°)")
        self.qtm_angle_canvas.axes.set_ylabel("QTM Angle (°)")
        self.joint_moment_canvas.axes.set_ylabel("Joint Moment (Nm/kg)")
        self.joint_moment_canvas.axes.set_xlabel("Time (s)")

        # Enable grid
        self.insole_canvas.axes.grid(True)
        self.qtm_force_canvas.axes.grid(True)
        self.opencap_angle_canvas.axes.grid(True)
        self.qtm_angle_canvas.axes.grid(True)
        self.joint_moment_canvas.axes.grid(True)

        # Tight layout
        for canvas in [
            self.insole_canvas,
            self.qtm_force_canvas,
            self.opencap_angle_canvas,
            self.qtm_angle_canvas,
            self.joint_moment_canvas,
        ]:
            canvas.fig.tight_layout()


class ComparisonPlotWindow(QMainWindow):
    """Window for displaying comparison plots of multiple data series"""
    def __init__(self, title, parent=None):
        super(ComparisonPlotWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create matplotlib canvas for the plot
        self.canvas = MatplotlibCanvas(width=8, height=6, dpi=100)
        main_layout.addWidget(self.canvas)
        
        # Initialize the figure
        self.canvas.axes.grid(True)
        self.canvas.axes.set_xlabel("Time (s)")
        self.canvas.axes.set_ylabel("Value")
        
    def plot_data_series(self, data_series, start_time, end_time):
        """Plot the selected data series"""
        self.canvas.axes.clear()
        self.canvas.axes.grid(True)
        
        legend_items = []
        
        # Plot each data series
        for name, time, values, color in data_series:
            # Filter data to the selected time range
            mask = (time >= start_time) & (time <= end_time)
            filtered_time = time[mask]
            filtered_values = values[mask]
            
            # Plot the data
            line, = self.canvas.axes.plot(filtered_time, filtered_values, 
                                         label=name, linewidth=1.5, color=color)
            legend_items.append(line)
        
        # Set the axis limits
        self.canvas.axes.set_xlim(start_time, end_time)
        
        # Add legend
        if legend_items:
            self.canvas.axes.legend(loc='upper right')
        
        # Update the canvas
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class QtSynchronizedVideoPlayer(QMainWindow):
    """
    Qt-based implementation of synchronized video and data visualization
    """

    def __init__(
        self,
        video_path,
        insole_data,
        qtm_force_data,
        opencap_joint_data,
        qtm_joint_data,
        insole_offset=DEFAULT_INSOLE_OFFSET,
        qtm_force_offset=DEFAULT_QTM_FORCE_OFFSET,
        opencap_knee_offset=DEFAULT_OPENCAP_KNEE_OFFSET,
        qtm_knee_offset=DEFAULT_QTM_KNEE_OFFSET,
    ):
        super(QtSynchronizedVideoPlayer, self).__init__()

        # Data initialization
        self.video_path = video_path
        self.insole_data = insole_data
        self.qtm_force_data = qtm_force_data
        self.opencap_joint_data = opencap_joint_data
        self.qtm_joint_data = qtm_joint_data

        # Initialize time offsets for each data stream (in seconds)
        self.insole_offset = insole_offset
        self.qtm_force_offset = qtm_force_offset
        self.opencap_joint_offset = opencap_knee_offset
        self.qtm_joint_offset = qtm_knee_offset

        # Initialize selected parameters
        self.selected_opencap_param = None
        self.selected_qtm_param = None
        self.selected_joint_moment = None
        self.subject_mass = 70.0  # Default subject mass in kg

        # Extract available parameters for dropdowns
        self.opencap_params = []
        if self.opencap_joint_data is not None:
            self.opencap_params = [
                col
                for col in self.opencap_joint_data.columns
                if col != "time" and not col.startswith("Time")
            ]
            if self.opencap_params:
                self.opencap_params = sorted(self.opencap_params)
                self.selected_opencap_param = self.opencap_params[0]

        self.qtm_params = []
        if self.qtm_joint_data is not None:
            self.qtm_params = [
                col
                for col in self.qtm_joint_data.columns
                if col != "TimeSeconds" and not col.startswith("Time")
            ]
            if self.qtm_params:
                self.qtm_params = sorted(self.qtm_params)
                self.selected_qtm_param = self.qtm_params[0]

        # Extract available joint moment parameters
        self.joint_moment_params = []
        if self.qtm_force_data is not None and self.qtm_joint_data is not None:
            self.joint_moment_params = [
                "Ankle_Moment",
                "Knee_Moment",
                "Hip_Moment"
            ]
            if self.joint_moment_params:
                self.selected_joint_moment = self.joint_moment_params[0]

        # Initialize video capture
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
        if self.opencap_joint_data is not None:
            data_frames.append((self.opencap_joint_data, "time"))
        if self.qtm_joint_data is not None:
            data_frames.append((self.qtm_joint_data, "TimeSeconds"))

        self.sync_frames, self.common_time = synchronize_data(self.fps, *data_frames)

        # Thread safety
        self.mutex = QMutex()
        self.is_playing = False

        # Setup UI
        self._setup_ui()

        # Update plots with initial data
        self.update_plots()

        # Initialize video timer
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.advance_frame)

        # Setup keyboard shortcuts
        self._setup_shortcuts()

    def _setup_ui(self):
        """Setup the user interface"""
        # Set window properties
        self.setWindowTitle("Synchronized Biomechanical Video Player")
        self.setMinimumSize(1200, 800)

        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Create a splitter for video and plots
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Create vertical splitter for video and controls
        vertical_splitter = QSplitter(Qt.Vertical)
        left_layout.addWidget(vertical_splitter)

        # Video container widget
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Video display
        self.video_frame = VideoFrame()
        video_layout.addWidget(self.video_frame)

        # Controls container widget
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)

        # Time slider
        slider_layout = QHBoxLayout()
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, self.frame_count)
        self.time_slider.setValue(0)

        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setMinimumWidth(140)

        slider_layout.addWidget(self.time_slider)
        slider_layout.addWidget(self.time_label)
        controls_layout.addLayout(slider_layout)

        # Playback controls
        playback_layout = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.prev_frame_button = QPushButton("←")
        self.next_frame_button = QPushButton("→")

        playback_layout.addWidget(self.prev_frame_button)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.next_frame_button)

        controls_layout.addLayout(playback_layout)

        # Time window controls
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Start Time (s):"))
        self.start_time_edit = QLineEdit("0.00")
        self.start_time_edit.setMaximumWidth(60)
        window_layout.addWidget(self.start_time_edit)

        window_layout.addWidget(QLabel("End Time (s):"))
        self.end_time_edit = QLineEdit(f"{self.duration:.2f}")
        self.end_time_edit.setMaximumWidth(60)
        window_layout.addWidget(self.end_time_edit)

        self.apply_window_button = QPushButton("Apply Window")
        window_layout.addWidget(self.apply_window_button)

        controls_layout.addLayout(window_layout)
        controls_layout.addStretch()

        # Add video and controls to vertical splitter
        vertical_splitter.addWidget(video_container)
        vertical_splitter.addWidget(controls_container)

        # Set initial sizes for vertical splitter
        vertical_splitter.setSizes([700, 200])

        # Middle panel - data plots
        self.plot_widget = DataPlotWidget()

        # Right panel - parameter selection and offset controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create scroll area for the controls (in case of many controls)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # 1. Insole Force Controls
        insole_group = QGroupBox("Insole Force Chart")
        insole_layout = QVBoxLayout(insole_group)

        # No parameter selection for insole force, just offset
        insole_offset_group = self._create_offset_group(
            "Time Offset", self.insole_offset, "insole"
        )
        insole_layout.addWidget(insole_offset_group)
        scroll_layout.addWidget(insole_group)

        # 2. QTM Force Controls
        qtm_force_group = QGroupBox("QTM Force Chart")
        qtm_force_layout = QVBoxLayout(qtm_force_group)

        # No parameter selection for QTM force, just offset
        qtm_force_offset_group = self._create_offset_group(
            "Time Offset", self.qtm_force_offset, "qtm_force"
        )
        qtm_force_layout.addWidget(qtm_force_offset_group)
        scroll_layout.addWidget(qtm_force_group)

        # 3. OpenCap Kinematics Controls
        opencap_group = QGroupBox("OpenCap Kinematics Chart")
        opencap_layout = QVBoxLayout(opencap_group)

        # Parameter selection for OpenCap
        opencap_param_layout = QGridLayout()
        opencap_param_layout.addWidget(QLabel("Joint Parameter:"), 0, 0)
        self.opencap_dropdown = QComboBox()
        if self.opencap_params:
            self.opencap_dropdown.addItems(self.opencap_params)
        opencap_param_layout.addWidget(self.opencap_dropdown, 0, 1)

        opencap_layout.addLayout(opencap_param_layout)

        # Offset control for OpenCap
        opencap_offset_group = self._create_offset_group(
            "Time Offset", self.opencap_joint_offset, "opencap"
        )
        opencap_layout.addWidget(opencap_offset_group)
        scroll_layout.addWidget(opencap_group)

        # 4. QTM Kinematics Controls
        qtm_kin_group = QGroupBox("QTM Kinematics Chart")
        qtm_kin_layout = QVBoxLayout(qtm_kin_group)

        # Parameter selection for QTM
        qtm_param_layout = QGridLayout()
        qtm_param_layout.addWidget(QLabel("Joint Parameter:"), 0, 0)
        self.qtm_dropdown = QComboBox()
        if self.qtm_params:
            self.qtm_dropdown.addItems(self.qtm_params)
        qtm_param_layout.addWidget(self.qtm_dropdown, 0, 1)

        qtm_kin_layout.addLayout(qtm_param_layout)

        # Offset control for QTM
        qtm_knee_offset_group = self._create_offset_group(
            "Time Offset", self.qtm_joint_offset, "qtm_knee"
        )
        qtm_kin_layout.addWidget(qtm_knee_offset_group)
        scroll_layout.addWidget(qtm_kin_group)
        
        # 5. Joint Moment Controls
        joint_moment_group = QGroupBox("Joint Moment Chart")
        joint_moment_layout = QVBoxLayout(joint_moment_group)
        
        # Parameter selection for joint moments
        joint_moment_param_layout = QGridLayout()
        joint_moment_param_layout.addWidget(QLabel("Joint:"), 0, 0)
        self.joint_moment_dropdown = QComboBox()
        if self.joint_moment_params:
            self.joint_moment_dropdown.addItems(self.joint_moment_params)
        joint_moment_param_layout.addWidget(self.joint_moment_dropdown, 0, 1)
        
        # Subject mass for normalization
        joint_moment_param_layout.addWidget(QLabel("Subject Mass (kg):"), 1, 0)
        self.subject_mass_edit = QLineEdit(f"{self.subject_mass:.1f}")
        self.subject_mass_edit.setMaximumWidth(60)
        joint_moment_param_layout.addWidget(self.subject_mass_edit, 1, 1)
        
        joint_moment_layout.addLayout(joint_moment_param_layout)
        scroll_layout.addWidget(joint_moment_group)
        
        # 6. Comparison Plot Controls
        comparison_group = QGroupBox("Comparison Plot")
        comparison_layout = QVBoxLayout(comparison_group)
        
        # Time range selection
        time_range_layout = QGridLayout()
        time_range_layout.addWidget(QLabel("Start Time (s):"), 0, 0)
        self.comparison_start_time = QLineEdit("0.00")
        self.comparison_start_time.setMaximumWidth(60)
        time_range_layout.addWidget(self.comparison_start_time, 0, 1)
        
        time_range_layout.addWidget(QLabel("End Time (s):"), 0, 2)
        self.comparison_end_time = QLineEdit(f"{self.duration:.2f}")
        self.comparison_end_time.setMaximumWidth(60)
        time_range_layout.addWidget(self.comparison_end_time, 0, 3)
        
        comparison_layout.addLayout(time_range_layout)
        
        # Data series selection
        series_selection_layout = QVBoxLayout()
        series_selection_layout.addWidget(QLabel("Select Data Series:"))
        
        # Insole Force series
        if self.insole_data is not None:
            self.insole_left_check = QCheckBox("Insole Left Force")
            self.insole_right_check = QCheckBox("Insole Right Force")
            series_selection_layout.addWidget(self.insole_left_check)
            series_selection_layout.addWidget(self.insole_right_check)
        
        # QTM Force series
        if self.qtm_force_data is not None:
            self.qtm_force_check = QCheckBox("QTM Force")
            series_selection_layout.addWidget(self.qtm_force_check)
        
        # OpenCap joint angles
        if self.opencap_joint_data is not None:
            opencap_group_box = QGroupBox("OpenCap Joints")
            opencap_series_layout = QVBoxLayout(opencap_group_box)
            
            self.opencap_checks = {}
            for param in self.opencap_params:
                check = QCheckBox(param.replace("_", " ").title())
                self.opencap_checks[param] = check
                opencap_series_layout.addWidget(check)
            
            series_selection_layout.addWidget(opencap_group_box)
        
        # QTM joint angles
        if self.qtm_joint_data is not None:
            qtm_group_box = QGroupBox("QTM Joints")
            qtm_series_layout = QVBoxLayout(qtm_group_box)
            
            self.qtm_checks = {}
            for param in self.qtm_params:
                check = QCheckBox(param.replace("_", " ").title())
                self.qtm_checks[param] = check
                qtm_series_layout.addWidget(check)
            
            series_selection_layout.addWidget(qtm_group_box)
        
        # Update comparison plot to include joint moments
        # Add joint moment series to comparison plot
        if self.joint_moment_params:
            joint_moment_group_box = QGroupBox("Joint Moments")
            joint_moment_series_layout = QVBoxLayout(joint_moment_group_box)
            
            self.joint_moment_checks = {}
            for param in self.joint_moment_params:
                check = QCheckBox(param.replace("_", " ").title())
                self.joint_moment_checks[param] = check
                joint_moment_series_layout.addWidget(check)
            
            series_selection_layout.addWidget(joint_moment_group_box)
        
        # Plot button
        plot_button_layout = QHBoxLayout()
        self.plot_comparison_button = QPushButton("Plot Comparison")
        plot_button_layout.addWidget(self.plot_comparison_button)

        # Add correlation button
        self.plot_correlation_button = QPushButton("Plot Correlation")
        plot_button_layout.addWidget(self.plot_correlation_button)

        # ML dataset generation button
        self.generate_dataset_button = QPushButton("Generate ML Dataset")
        plot_button_layout.addWidget(self.generate_dataset_button)

        plot_button_layout.addStretch()
        
        comparison_layout.addLayout(series_selection_layout)
        comparison_layout.addLayout(plot_button_layout)
        
        # Add comparison group to scroll layout
        scroll_layout.addWidget(comparison_group)

        # Add ML dataset generation controls
        ml_dataset_group = QGroupBox("ML Dataset Generation")
        ml_dataset_layout = QVBoxLayout(ml_dataset_group)
        
        # Description label
        ml_description = QLabel(
            "Generate a dataset to train an ML model that predicts joint moments\n"
            "using OpenGo force and OpenCap kinematics as input features."
        )
        ml_description.setWordWrap(True)
        ml_dataset_layout.addWidget(ml_description)
        
        # Create dataset button
        self.generate_dataset_button = QPushButton("Generate ML Dataset")
        ml_dataset_layout.addWidget(self.generate_dataset_button)
        
        # Add ML dataset group to scroll layout
        scroll_layout.addWidget(ml_dataset_group)

        # Add stretch to push controls to the top
        scroll_layout.addStretch()

        # Set the scroll content
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(right_panel)

        # Set initial splitter sizes (ratios)
        splitter.setSizes([300, 600, 300])

        # Connect signals
        self._connect_signals()

        # Initialize UI state
        self._update_time_display()

        # Set initial frame
        self.update_frame()

    def _connect_signals(self):
        """Connect UI signals to handlers"""
        # Playback controls
        self.play_button.clicked.connect(self.toggle_play)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.next_frame_button.clicked.connect(self.next_frame)

        # Time slider
        self.time_slider.valueChanged.connect(self.on_slider_change)

        # Window controls
        self.apply_window_button.clicked.connect(self.on_window_change)

        # Parameter selection
        self.opencap_dropdown.currentTextChanged.connect(self.on_opencap_param_change)
        self.qtm_dropdown.currentTextChanged.connect(self.on_qtm_param_change)
        self.joint_moment_dropdown.currentTextChanged.connect(self.on_joint_moment_change)
        
        # Subject mass
        self.subject_mass_edit.editingFinished.connect(self.on_subject_mass_change)
        
        # Comparison plot button
        self.plot_comparison_button.clicked.connect(self.on_plot_comparison)

        # Correlation plot button
        self.plot_correlation_button.clicked.connect(self.on_plot_correlation)

        # ML dataset generation button
        self.generate_dataset_button.clicked.connect(self.on_generate_ml_dataset)

        # Store references to the offset sliders to connect their signals
        # The signals from the offset groups are connected when they are created
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for the application"""
        # Space to play/pause
        self.shortcut_space = create_shortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.activated.connect(self.toggle_play)

        # Left/right arrows to move frame by frame
        self.shortcut_left = create_shortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.prev_frame)

        self.shortcut_right = create_shortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.next_frame)
    
    def on_slider_change(self, value):
        """Handle time slider value change"""
        if value != self.current_frame:
            self.current_frame = value
            self.update_frame()

    def on_window_change(self):
        """Handle time window change"""
        try:
            start_time = float(self.start_time_edit.text())
            end_time = float(self.end_time_edit.text())

            # Validate window
            start_time = max(0.0, start_time)
            end_time = min(self.duration, end_time)

            if start_time >= end_time:
                end_time = start_time + 0.1
                end_time = min(self.duration, end_time)

            # Update UI
            self.start_time_edit.setText(f"{start_time:.2f}")
            self.end_time_edit.setText(f"{end_time:.2f}")

            # Update plots
            self.time_window_start = start_time
            self.time_window_end = end_time
            self.update_plots()

        except ValueError:
            # Restore default if invalid input
            self.start_time_edit.setText("0.00")
            self.end_time_edit.setText(f"{self.duration:.2f}")

    def on_opencap_param_change(self, param):
        """Handle OpenCap parameter selection change"""
        self.selected_opencap_param = param
        self.update_plots()

    def on_qtm_param_change(self, param):
        """Handle QTM parameter selection change"""
        self.selected_qtm_param = param
        self.update_plots()

    def on_joint_moment_change(self, param):
        """Handle joint moment parameter selection change"""
        self.selected_joint_moment = param
        self.update_plots()

    def on_subject_mass_change(self):
        """Handle subject mass change"""
        try:
            mass = float(self.subject_mass_edit.text())
            if mass > 0:
                self.subject_mass = mass
                self.update_plots()
            else:
                # Reset to previous value if invalid
                self.subject_mass_edit.setText(f"{self.subject_mass:.1f}")
        except ValueError:
            # Reset to previous value if invalid
            self.subject_mass_edit.setText(f"{self.subject_mass:.1f}")

    def on_offset_change(self, stream_id, value):
        """Handle offset change for any data stream"""
        if stream_id == "insole":
            self.insole_offset = value
        elif stream_id == "qtm_force":
            self.qtm_force_offset = value
        elif stream_id == "opencap":
            self.opencap_joint_offset = value
        elif stream_id == "qtm_knee":
            self.qtm_joint_offset = value

        self.update_plots()

    def _update_time_display(self):
        """Update the time display label"""
        current_time = self.current_frame / self.fps
        minutes_current = int(current_time // 60)
        seconds_current = current_time % 60

        total_minutes = int(self.duration // 60)
        total_seconds = self.duration % 60

        self.time_label.setText(
            f"{minutes_current}:{seconds_current:05.2f} / "
            f"{total_minutes}:{total_seconds:05.2f}"
        )

    def update_frame(self):
        """Update the video frame display"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if ret:
            # Update video display
            self.video_frame.set_frame(frame)

            # Update slider position
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(self.current_frame)
            self.time_slider.blockSignals(False)

            # Update time display
            self._update_time_display()

            # Update cursor position in plots
            self.update_cursor()

    def update_cursor(self):
        """Update cursor position in plots"""
        current_time = self.current_frame / self.fps

        # Remove old cursor lines
        while self.plot_widget.cursor_lines:
            line = self.plot_widget.cursor_lines.pop()
            try:
                line.remove()
            except:
                pass

        # Create new cursor lines
        for axes in [
            self.plot_widget.insole_canvas.axes,
            self.plot_widget.qtm_force_canvas.axes,
            self.plot_widget.opencap_angle_canvas.axes,
            self.plot_widget.qtm_angle_canvas.axes,
            self.plot_widget.joint_moment_canvas.axes,
        ]:
            cursor_line = axes.axvline(
                current_time, color="black", linestyle="--", linewidth=0.8
            )
            self.plot_widget.cursor_lines.append(cursor_line)

        # Redraw canvases
        for canvas in [
            self.plot_widget.insole_canvas,
            self.plot_widget.qtm_force_canvas,
            self.plot_widget.opencap_angle_canvas,
            self.plot_widget.qtm_angle_canvas,
            self.plot_widget.joint_moment_canvas,
        ]:
            canvas.draw()

    def update_plots(self):
        """Update all data plots with current settings"""
        # Clear all existing plots
        for canvas in [
            self.plot_widget.insole_canvas,
            self.plot_widget.qtm_force_canvas,
            self.plot_widget.opencap_angle_canvas,
            self.plot_widget.qtm_angle_canvas,
            self.plot_widget.joint_moment_canvas,
        ]:
            canvas.axes.clear()
            canvas.axes.grid(True)

        # Reset plot data
        self.plot_widget.insole_lines = []
        self.plot_widget.qtm_force_lines = []
        self.plot_widget.opencap_angle_lines = []
        self.plot_widget.qtm_angle_lines = []
        self.plot_widget.joint_moment_lines = []
        self.plot_widget.cursor_lines = []

        # Plot index counter
        idx = 0

        # Track sync frames for each data type
        insole_sync = None
        qtm_force_sync = None
        opencap_joint_sync = None
        qtm_joint_sync = None

        # Insole force plot
        if self.insole_data is not None:
            insole_sync = self.sync_frames[idx]
            idx += 1

            # Apply filtering to smooth the signals
            left_force = filter_signal(insole_sync["Left_Force"].values)
            right_force = filter_signal(insole_sync["Right_Force"].values)

            # Apply offset to time values
            adjusted_time = insole_sync["time"] + self.insole_offset

            # Plot the data
            (left_line,) = self.plot_widget.insole_canvas.axes.plot(
                adjusted_time, left_force, "g-", label="Left Force", linewidth=0.8
            )
            (right_line,) = self.plot_widget.insole_canvas.axes.plot(
                adjusted_time, right_force, "r-", label="Right Force", linewidth=0.8
            )

            self.plot_widget.insole_lines.extend([left_line, right_line])
            self.plot_widget.insole_canvas.axes.set_ylabel("Insole Force (N)")
            self.plot_widget.insole_canvas.axes.legend(loc="upper right")

        # QTM force plot
        if self.qtm_force_data is not None:
            qtm_force_sync = self.sync_frames[idx]
            idx += 1

            # Apply filtering
            force = filter_signal(qtm_force_sync["Force"].values)

            # Apply offset to time values
            adjusted_time = qtm_force_sync["time"] + self.qtm_force_offset

            # Plot the data
            (force_line,) = self.plot_widget.qtm_force_canvas.axes.plot(
                adjusted_time, force, "b-", label="Force", linewidth=0.8
            )

            self.plot_widget.qtm_force_lines.append(force_line)
            self.plot_widget.qtm_force_canvas.axes.set_ylabel("QTM Force (N)")
            self.plot_widget.qtm_force_canvas.axes.legend(loc="upper right")

        # OpenCap joint angles plot
        if self.opencap_joint_data is not None and self.selected_opencap_param:
            opencap_joint_sync = self.sync_frames[idx]
            idx += 1

            # Apply offset to time values
            adjusted_time = opencap_joint_sync["time"] + self.opencap_joint_offset

            # Plot only the selected parameter
            if self.selected_opencap_param in opencap_joint_sync.columns:
                # Apply filtering
                angles = filter_signal(
                    opencap_joint_sync[self.selected_opencap_param].values
                )

                label = self.selected_opencap_param.replace("_", " ").title()
                (angle_line,) = self.plot_widget.opencap_angle_canvas.axes.plot(
                    adjusted_time, angles, label=label, linewidth=1.2, color="blue"
                )

                self.plot_widget.opencap_angle_lines.append(angle_line)
                self.plot_widget.opencap_angle_canvas.axes.set_ylabel(
                    f"OpenCap {label} (°)"
                )
                self.plot_widget.opencap_angle_canvas.axes.legend(loc="upper right")

        # QTM joint angles plot
        if self.qtm_joint_data is not None and self.selected_qtm_param:
            qtm_joint_sync = self.sync_frames[idx]
            idx += 1

            # Apply offset to time values
            adjusted_time = qtm_joint_sync["time"] + self.qtm_joint_offset

            # Plot only the selected parameter
            if self.selected_qtm_param in qtm_joint_sync.columns:
                # Apply filtering
                angles = filter_signal(qtm_joint_sync[self.selected_qtm_param].values)

                # Only apply the 180 degree adjustment for knee angles
                if "knee" in self.selected_qtm_param.lower():
                    angles = -angles + 180  # Invert direction to match OpenCap convention

                label = self.selected_qtm_param.replace("_", " ").title()
                (angle_line,) = self.plot_widget.qtm_angle_canvas.axes.plot(
                    adjusted_time, angles, label=label, linewidth=1.2, color="red"
                )

                self.plot_widget.qtm_angle_lines.append(angle_line)
                self.plot_widget.qtm_angle_canvas.axes.set_ylabel(f"QTM {label} (°)")
                self.plot_widget.qtm_angle_canvas.axes.legend(loc="upper right")

        # Joint moment plot
        if (self.qtm_force_data is not None and 
            self.qtm_joint_data is not None and 
            self.selected_joint_moment and 
            self.selected_qtm_param):
            
            # Make sure we found the sync frames for both data sources
            if qtm_force_sync is not None and qtm_joint_sync is not None:
                try:
                    print(f"Attempting to calculate {self.selected_joint_moment} using {self.selected_qtm_param} angles")
                    
                    # Get force data and ensure it's not empty
                    forces = np.array(qtm_force_sync["Force"].values)
                    
                    # Get angle data for the selected joint
                    angles = np.array(qtm_joint_sync[self.selected_qtm_param].values)
                    
                    non_nan_forces = forces[~np.isnan(forces)]
                    non_nan_angles = angles[~np.isnan(angles)]
                    
                    print(f"Force data shape: {forces.shape}, valid values: {len(non_nan_forces)}, range: {np.min(non_nan_forces) if len(non_nan_forces) > 0 else 'N/A'}-{np.max(non_nan_forces) if len(non_nan_forces) > 0 else 'N/A'}")
                    print(f"Angle data shape: {angles.shape}, valid values: {len(non_nan_angles)}, range: {np.min(non_nan_angles) if len(non_nan_angles) > 0 else 'N/A'}-{np.max(non_nan_angles) if len(non_nan_angles) > 0 else 'N/A'}")
                    
                    # Adjust for knee angles if needed
                    if "knee" in self.selected_qtm_param.lower():
                        # Use np.nan_to_num to handle NaN values during the calculation
                        angles = -np.nan_to_num(angles, nan=0.0) + 180
                        print(f"Adjusted knee angles, new range: {np.min(angles[~np.isnan(angles)]) if np.sum(~np.isnan(angles)) > 0 else 'N/A'}-{np.max(angles[~np.isnan(angles)]) if np.sum(~np.isnan(angles)) > 0 else 'N/A'}")
                    
                    # Check data lengths and resample if necessary
                    if len(forces) != len(angles):
                        # Use the shorter length
                        min_len = min(len(forces), len(angles))
                        forces = forces[:min_len]
                        angles = angles[:min_len]
                        print(f"Resampled data to common length: {min_len}")
                    
                    # Safety check
                    if len(forces) == 0 or len(angles) == 0:
                        raise ValueError("Empty data arrays after preprocessing")
                        
                    # Set an appropriate lever arm length based on the selected joint
                    if "ankle" in self.selected_joint_moment.lower():
                        lever_arm_factor = 0.05  # Ankle has shorter moment arm
                    elif "knee" in self.selected_joint_moment.lower():
                        lever_arm_factor = 0.25  # Knee has medium moment arm
                    elif "hip" in self.selected_joint_moment.lower():
                        lever_arm_factor = 0.40  # Hip has longer moment arm
                    else:
                        lever_arm_factor = 0.25  # Default
                        
                    # Calculate joint moments
                    moments = calculate_joint_moments(forces, angles, self.subject_mass)
                    
                    if len(moments) == 0:
                        raise ValueError("Joint moment calculation returned empty array")
                        
                    # Copy time array and ensure it has the right length
                    times = qtm_force_sync["time"].values.copy()
                    if len(times) > len(moments):
                        times = times[:len(moments)]
                    elif len(times) < len(moments):
                        # In case we need to extend the time array
                        time_step = times[1] - times[0] if len(times) > 1 else 1/self.fps
                        extra_times = np.arange(len(times), len(moments)) * time_step + times[-1]
                        times = np.concatenate([times, extra_times])
                    
                    # Apply offset to time values
                    adjusted_time = times + self.qtm_force_offset
                    
                    print(f"Generated moment data: {len(moments)} points, time: {len(adjusted_time)} points")
                    
                    # Verify the data is valid for plotting
                    if len(moments) > 0 and not np.all(np.isnan(moments)):
                        label = self.selected_joint_moment.replace("_", " ").title()
                        
                        # Make sure moments are finite for plotting
                        plotable_moments = np.nan_to_num(moments, nan=0.0)
                        
                        # Plot the moments
                        (moment_line,) = self.plot_widget.joint_moment_canvas.axes.plot(
                            adjusted_time, plotable_moments, label=label, linewidth=1.2, color="purple"
                        )
                        
                        self.plot_widget.joint_moment_lines.append(moment_line)
                        self.plot_widget.joint_moment_canvas.axes.set_ylabel(f"Joint Moment (Nm/kg)")
                        self.plot_widget.joint_moment_canvas.axes.legend(loc="upper right")
                    else:
                        raise ValueError("No valid moment data for plotting")
                except Exception as e:
                    print(f"Error plotting joint moments: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Create a text annotation to show in the plot
                    self.plot_widget.joint_moment_canvas.axes.text(
                        0.5, 0.5, 
                        f"Could not calculate moments:\n{str(e)}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=self.plot_widget.joint_moment_canvas.axes.transAxes
                    )

        # Set x-axis label and limits
        self.plot_widget.qtm_angle_canvas.axes.set_xlabel("Time (s)")

        # Set common time window for all plots including joint moment
        time_window_start = (
            float(self.start_time_edit.text())
            if hasattr(self, "start_time_edit")
            else 0.0
        )
        time_window_end = (
            float(self.end_time_edit.text())
            if hasattr(self, "end_time_edit")
            else self.duration
        )

        for axes in [
            self.plot_widget.insole_canvas.axes,
            self.plot_widget.qtm_force_canvas.axes,
            self.plot_widget.opencap_angle_canvas.axes,
            self.plot_widget.qtm_angle_canvas.axes,
            self.plot_widget.joint_moment_canvas.axes,
        ]:
            axes.set_xlim(time_window_start, time_window_end)

        # Update the cursor
        self.update_cursor()

        # Apply tight layout to each figure including joint moment
        for canvas in [
            self.plot_widget.insole_canvas,
            self.plot_widget.qtm_force_canvas,
            self.plot_widget.opencap_angle_canvas,
            self.plot_widget.qtm_angle_canvas,
            self.plot_widget.joint_moment_canvas,
        ]:
            canvas.fig.tight_layout()
            canvas.draw()

    def toggle_play(self):
        """Toggle video playback"""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start video playback"""
        self.is_playing = True
        self.play_button.setText("Pause")

        # Start the timer with an interval based on fps
        interval = int(1000 / self.fps)
        self.play_timer.start(interval)

    def pause(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_button.setText("Play")
        self.play_timer.stop()

    def advance_frame(self):
        """Advance to the next frame during playback"""
        self.mutex.lock()
        self.current_frame += 1

        # Check if we've reached the end
        if self.current_frame >= self.frame_count:
            self.current_frame = self.frame_count - 1
            self.pause()

        self.update_frame()
        self.mutex.unlock()

    def prev_frame(self):
        """Go to previous frame"""
        self.mutex.lock()
        self.current_frame = max(0, self.current_frame - 1)
        self.update_frame()
        self.mutex.unlock()

    def next_frame(self):
        """Go to next frame"""
        self.mutex.lock()
        self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
        self.update_frame()
        self.mutex.unlock()

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop playback timer
        self.play_timer.stop()

        # Release video capture
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        # Accept the close event
        event.accept()

    def _create_offset_group(self, title, default_value, stream_id):
        """Create a group of controls for a data stream offset"""
        container = QWidget()
        layout = QGridLayout(container)

        # Slider for coarse adjustment
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-500, 500)  # Range from -5.0 to 5.0 seconds (x100)
        slider.setValue(int(default_value * 100))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(100)

        # Text input for precise adjustment
        value_edit = QLineEdit(f"{default_value:.2f}")
        value_edit.setMaximumWidth(60)

        # Add widgets to layout
        layout.addWidget(QLabel("Offset (s):"), 0, 0)
        layout.addWidget(slider, 0, 1)
        layout.addWidget(value_edit, 0, 2)

        # Connect signals
        def on_slider_change(value):
            float_value = value / 100.0
            value_edit.setText(f"{float_value:.2f}")
            self.on_offset_change(stream_id, float_value)

        def on_edit_change():
            try:
                float_value = float(value_edit.text())
                if -5.0 <= float_value <= 5.0:
                    slider.setValue(int(float_value * 100))
                    self.on_offset_change(stream_id, float_value)
            except ValueError:
                # Restore previous value
                value_edit.setText(f"{slider.value() / 100.0:.2f}")

        slider.valueChanged.connect(on_slider_change)
        value_edit.editingFinished.connect(on_edit_change)

        return container

    def on_plot_comparison(self):
        """Handle plot comparison button click"""
        try:
            # Get time range
            start_time = float(self.comparison_start_time.text())
            end_time = float(self.comparison_end_time.text())
            
            # Validate time range
            start_time = max(0.0, start_time)
            end_time = min(self.duration, end_time)
            
            if start_time >= end_time:
                end_time = start_time + 0.1
                end_time = min(self.duration, end_time)
            
            # Collect selected data series
            data_series = []
            
            # Insole data
            if self.insole_data is not None:
                insole_sync = self.sync_frames[0]
                adjusted_time = insole_sync["time"] + self.insole_offset
                
                if hasattr(self, 'insole_left_check') and self.insole_left_check.isChecked():
                    left_force = filter_signal(insole_sync["Left_Force"].values)
                    data_series.append(("Insole Left Force", adjusted_time, left_force, 'green'))
                
                if hasattr(self, 'insole_right_check') and self.insole_right_check.isChecked():
                    right_force = filter_signal(insole_sync["Right_Force"].values)
                    data_series.append(("Insole Right Force", adjusted_time, right_force, 'red'))
            
            # QTM force data
            if self.qtm_force_data is not None:
                idx = 1 if self.insole_data is not None else 0
                qtm_force_sync = self.sync_frames[idx]
                adjusted_time = qtm_force_sync["time"] + self.qtm_force_offset
                
                if hasattr(self, 'qtm_force_check') and self.qtm_force_check.isChecked():
                    force = filter_signal(qtm_force_sync["Force"].values)
                    data_series.append(("QTM Force", adjusted_time, force, 'blue'))
            
            # OpenCap joint data
            if self.opencap_joint_data is not None:
                idx = (1 if self.insole_data is not None else 0) + (1 if self.qtm_force_data is not None else 0)
                opencap_joint_sync = self.sync_frames[idx]
                adjusted_time = opencap_joint_sync["time"] + self.opencap_joint_offset
                
                if hasattr(self, 'opencap_checks'):
                    for param, check in self.opencap_checks.items():
                        if check.isChecked() and param in opencap_joint_sync.columns:
                            angles = filter_signal(opencap_joint_sync[param].values)
                            label = "OpenCap " + param.replace("_", " ").title()
                            data_series.append((label, adjusted_time, angles, 'darkblue'))
            
            # QTM joint data
            if self.qtm_joint_data is not None:
                idx = (1 if self.insole_data is not None else 0) + \
                      (1 if self.qtm_force_data is not None else 0) + \
                      (1 if self.opencap_joint_data is not None else 0)
                qtm_joint_sync = self.sync_frames[idx]
                adjusted_time = qtm_joint_sync["time"] + self.qtm_joint_offset
                
                if hasattr(self, 'qtm_checks'):
                    for param, check in self.qtm_checks.items():
                        if check.isChecked() and param in qtm_joint_sync.columns:
                            angles = filter_signal(qtm_joint_sync[param].values)
                            
                            # Only apply the 180 degree adjustment for knee angles
                            if "knee" in param.lower():
                                angles = -angles + 180  # Invert direction to match OpenCap convention
                            
                            label = "QTM " + param.replace("_", " ").title()
                            data_series.append((label, adjusted_time, angles, 'darkred'))
            
            # Add Joint Moment data to comparison
            if self.qtm_force_data is not None and self.qtm_joint_data is not None:
                # Find indices for force and joint data
                force_idx = 1 if self.insole_data is not None else 0
                joint_idx = (1 if self.insole_data is not None else 0) + \
                            (1 if self.qtm_force_data is not None else 0) + \
                            (1 if self.opencap_joint_data is not None else 0)
                
                qtm_force_sync = self.sync_frames[force_idx]
                qtm_joint_sync = self.sync_frames[joint_idx]
                
                # Get adjusted time
                adjusted_time = qtm_force_sync["time"] + self.qtm_force_offset
                
                # Add moment data series if selected
                if hasattr(self, 'joint_moment_checks'):
                    for param, check in self.joint_moment_checks.items():
                        if check.isChecked():
                            # Get the associated joint angle parameter
                            angle_param = None
                            if "Ankle" in param and "Ankle" in self.selected_qtm_param:
                                angle_param = self.selected_qtm_param
                            elif "Knee" in param and "Knee" in self.selected_qtm_param:
                                angle_param = self.selected_qtm_param
                            elif "Hip" in param and "Hip" in self.selected_qtm_param:
                                angle_param = self.selected_qtm_param
                            
                            if angle_param is not None and angle_param in qtm_joint_sync.columns:
                                forces = qtm_force_sync["Force"].values
                                angles = qtm_joint_sync[angle_param].values
                                
                                # Adjust for knee angles if needed
                                if "knee" in angle_param.lower():
                                    angles = -angles + 180
                                
                                # Calculate moments
                                moments = calculate_joint_moments(
                                    forces.reshape(-1, 1),
                                    angles.reshape(-1, 1),
                                    self.subject_mass
                                )
                                
                                label = param.replace("_", " ").title()
                                data_series.append((label, adjusted_time, moments, 'purple'))

            # Create and show comparison plot window if we have data
            if data_series:
                window_title = f"Comparison Plot ({start_time:.2f}s - {end_time:.2f}s)"
                comparison_window = ComparisonPlotWindow(window_title, self)
                comparison_window.plot_data_series(data_series, start_time, end_time)
                comparison_window.show()
        
        except ValueError:
            # Reset to default values if there's an error
            self.comparison_start_time.setText("0.00")
            self.comparison_end_time.setText(f"{self.duration:.2f}")

    def on_plot_correlation(self):
        """Handle plot correlation button click"""
        try:
            # Get time range
            start_time = float(self.comparison_start_time.text())
            end_time = float(self.comparison_end_time.text())
            
            # Validate time range
            start_time = max(0.0, start_time)
            end_time = min(self.duration, end_time)
            
            if start_time >= end_time:
                end_time = start_time + 0.1
                end_time = min(self.duration, end_time)
            
            # Collect selected data from the UI
            selected_data = collect_selected_data(self)
            
            # Analyze correlation and show results
            analyze_correlation(self, selected_data, (start_time, end_time), self.subject_mass)
            
        except Exception as e:
            # Show error dialog
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Error in correlation analysis: {str(e)}")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            
            import traceback
            traceback.print_exc()

    def on_generate_ml_dataset(self):
        """Handle ML dataset generation button click"""
        # Import the ML dataset module and show the window
        from .ml_dataset_module import show_ml_dataset_window
        
        # Store a reference to prevent garbage collection
        self.ml_dataset_window = show_ml_dataset_window(self)


# Simplified shortcut class for backward compatibility
from PyQt5.QtWidgets import QShortcut as QtQShortcut


# Shortcut methods for keyboard interaction
def create_shortcut(keysequence, parent):
    """Create a PyQt keyboard shortcut"""
    return QtQShortcut(keysequence, parent)


# For backward compatibility with the original implementation
class SynchronizedVideoPlayer(QtSynchronizedVideoPlayer):
    """Alias class for backward compatibility"""

    pass
