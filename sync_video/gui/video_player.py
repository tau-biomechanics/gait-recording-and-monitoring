"""
Synchronized video player implementation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation
import matplotlib.lines as lines

from ..utils.signal_processing import filter_signal
from ..utils.synchronization import synchronize_data
from ..config import (
    DEFAULT_INSOLE_OFFSET,
    DEFAULT_QTM_FORCE_OFFSET,
    DEFAULT_OPENCAP_KNEE_OFFSET,
    DEFAULT_QTM_KNEE_OFFSET
)


from matplotlib.widgets import AxesWidget
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

# A proper scrollable dropdown menu for matplotlib
class ScrollableDropdown:
    """
    A scrollable dropdown menu for matplotlib that properly handles long lists
    """
    def __init__(self, ax, labels, active=0, callback=None, max_display=6):
        """
        Initialize a scrollable dropdown menu
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes instance to place the dropdown menu in
        labels : list of str
            The list of labels to display in the dropdown
        active : int, optional
            The index of the initially selected label
        callback : function, optional
            The function to call when a selection is made
        max_display : int, optional
            The maximum number of options to display at once in the dropdown
        """
        self.ax = ax
        self.fig = ax.figure
        self.labels = labels
        self.active = active
        self.callback = callback
        self.max_display = min(max_display, len(labels))  # Don't exceed list length
        
        # Setup main button appearance
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_axis_off()
        
        # Create dropdown button with gradient effect and arrow
        # Button border
        border = patches.FancyBboxPatch(
            (-0.01, -0.01), 1.02, 1.02, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='none', edgecolor='#4477AA', linewidth=1.5
        )
        self.ax.add_patch(border)
        
        # Button face
        self.button_rect = patches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='whitesmoke', edgecolor='none', alpha=0.9
        )
        self.ax.add_patch(self.button_rect)
        
        # Add dropdown arrow with better styling
        self.arrow = self.ax.text(
            0.95, 0.5, '▼', ha='center', va='center', 
            fontsize=10, color='#4477AA', weight='bold'
        )
        
        # Display the currently selected label
        self.text = self.ax.text(
            0.5, 0.5, self.labels[self.active], 
            ha='center', va='center', fontsize=9,
            weight='bold'
        )
        
        # Create a popup axes for the dropdown list
        self.popup_visible = False
        self.popup_ax = None
        self.option_rects = []
        self.option_texts = []
        
        # Scrolling state
        self.scroll_offset = 0
        self.scroll_up_button = None
        self.scroll_down_button = None
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        
    def _on_click(self, event):
        """Handle mouse click events for the dropdown"""
        # Handle click on the main dropdown button
        if not self.popup_visible and event.inaxes == self.ax:
            self._show_dropdown()
            return
            
        # If popup not showing, nothing else to do
        if not self.popup_visible:
            return
            
        # Check if clicked outside the dropdown and popup
        if event.inaxes != self.popup_ax and event.inaxes != self.ax:
            self._hide_dropdown()
            return
            
        # Check for clicks on scroll buttons
        if self.scroll_up_button and self.scroll_up_button.contains(event)[0]:
            self._scroll_up()
            return
            
        if self.scroll_down_button and self.scroll_down_button.contains(event)[0]:
            self._scroll_down()
            return
            
        # Check for clicks on options
        if event.inaxes == self.popup_ax:
            for i, rect in enumerate(self.option_rects):
                if rect.contains(event)[0]:
                    # Calculate the actual index in the full list
                    actual_idx = i + self.scroll_offset
                    if 0 <= actual_idx < len(self.labels):
                        self.set_active(actual_idx)
                        self._hide_dropdown()
                        return
    
    def _on_scroll(self, event):
        """Handle mouse scroll events when dropdown is open"""
        if not self.popup_visible or event.inaxes != self.popup_ax:
            return
            
        # Determine scroll direction
        if event.button == 'up':
            self._scroll_up()
        elif event.button == 'down':
            self._scroll_down()
            
        # Prevent further propagation
        return
    
    def _scroll_up(self):
        """Scroll the dropdown list up"""
        if self.scroll_offset > 0:
            self.scroll_offset -= 1
            self._update_dropdown_options()
            
    def _scroll_down(self):
        """Scroll the dropdown list down"""
        if self.scroll_offset < len(self.labels) - self.max_display:
            self.scroll_offset += 1
            self._update_dropdown_options()
    
    def _show_dropdown(self):
        """Show the dropdown list"""
        if self.popup_visible:
            return
            
        self.popup_visible = True
        
        # Change button appearance to "pressed" state
        self.button_rect.set_facecolor('lightblue')
        self.arrow.set_text('▲')
        
        # Create popup axes for dropdown list
        btn_pos = self.ax.get_position()
        popup_height = min(0.3, btn_pos.height * self.max_display)  # Limit height
        
        self.popup_ax = self.fig.add_axes([
            btn_pos.x0,  # Same x position as button
            btn_pos.y0 - popup_height,  # Position below button
            btn_pos.width,  # Same width as button
            popup_height  # Height based on number of visible options
        ], zorder=100)  # Higher zorder to ensure it appears on top
        
        self.popup_ax.set_xlim(0, 1)
        self.popup_ax.set_ylim(0, 1)
        self.popup_ax.set_axis_off()
        
        # Create background with drop shadow effect
        # First add shadow (slightly larger, darker box behind)
        shadow = patches.FancyBboxPatch(
            (0.01, -0.01), 1.0, 1.0, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='darkgray', edgecolor='gray', alpha=0.5,
            zorder=98
        )
        self.popup_ax.add_patch(shadow)
        
        # Main popup background
        self.popup_bg = patches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0, 
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor='white', edgecolor='#4477AA', alpha=0.97,
            linewidth=1.5, zorder=99
        )
        self.popup_ax.add_patch(self.popup_bg)
        
        # Reset scroll offset to ensure active item is visible
        visible_range = range(self.scroll_offset, self.scroll_offset + self.max_display)
        if self.active not in visible_range:
            # Try to center the active item in the visible range
            self.scroll_offset = max(0, min(self.active - self.max_display//2, 
                                          len(self.labels) - self.max_display))
        
        # Add scroll buttons if needed
        if len(self.labels) > self.max_display:
            # Up scroll button at top
            self.scroll_up_button = patches.Circle(
                (0.9, 0.95), 0.05, 
                facecolor='lightgray' if self.scroll_offset > 0 else 'whitesmoke',
                alpha=0.8
            )
            self.popup_ax.add_patch(self.scroll_up_button)
            self.popup_ax.text(0.9, 0.95, '▲', ha='center', va='center', fontsize=8)
            
            # Down scroll button at bottom
            self.scroll_down_button = patches.Circle(
                (0.9, 0.05), 0.05, 
                facecolor='lightgray' if self.scroll_offset < len(self.labels) - self.max_display else 'whitesmoke',
                alpha=0.8
            )
            self.popup_ax.add_patch(self.scroll_down_button)
            self.popup_ax.text(0.9, 0.05, '▼', ha='center', va='center', fontsize=8)
        
        # Add the options
        self._update_dropdown_options()
        
        # Draw the figure
        self.fig.canvas.draw_idle()
    
    def _update_dropdown_options(self):
        """Update the visible options in the dropdown"""
        # Clear existing options
        for text in self.option_texts:
            text.remove()
        for rect in self.option_rects:
            rect.remove()
            
        self.option_texts = []
        self.option_rects = []
        
        # Calculate height for each option
        option_height = 1.0 / self.max_display
        
        # Add visible options
        visible_end = min(self.scroll_offset + self.max_display, len(self.labels))
        
        for i in range(self.scroll_offset, visible_end):
            rel_idx = i - self.scroll_offset  # Relative index in visible list
            y_pos = 1.0 - (rel_idx + 1) * option_height
            
            # Option background
            is_active = (i == self.active)
            rect = patches.Rectangle(
                (0.05, y_pos), 0.8, option_height,
                facecolor='#BBDEFB' if is_active else 'white',  # Lighter blue for selected item
                edgecolor='#4477AA' if is_active else 'none',   # Border for selected item
                linewidth=1.0,
                alpha=0.95, picker=True
            )
            self.popup_ax.add_patch(rect)
            self.option_rects.append(rect)
            
            # Option text
            text = self.popup_ax.text(
                0.1, y_pos + option_height/2, self.labels[i],
                ha='left', va='center', fontsize=9,
                weight='bold' if is_active else 'normal',  # Bold for the active item
                color='#333333'
            )
            self.option_texts.append(text)
        
        # Update scroll button appearances
        if self.scroll_up_button:
            self.scroll_up_button.set_facecolor('lightgray' if self.scroll_offset > 0 else 'whitesmoke')
        
        if self.scroll_down_button:
            self.scroll_down_button.set_facecolor('lightgray' if self.scroll_offset < len(self.labels) - self.max_display else 'whitesmoke')
            
        # Draw the figure
        self.fig.canvas.draw_idle()
    
    def _hide_dropdown(self):
        """Hide the dropdown list"""
        if not self.popup_visible:
            return
            
        self.popup_visible = False
        
        # Reset button appearance
        self.button_rect.set_facecolor('whitesmoke')
        self.arrow.set_text('▼')
        
        # Remove popup axes
        if self.popup_ax:
            self.popup_ax.remove()
            self.popup_ax = None
            
        # Clear references
        self.option_rects = []
        self.option_texts = []
        self.scroll_up_button = None
        self.scroll_down_button = None
        
        # Draw the figure
        self.fig.canvas.draw_idle()
    
    def set_active(self, index):
        """Set the active selection"""
        if 0 <= index < len(self.labels):
            self.active = index
            self.text.set_text(self.labels[self.active])
            
            # Call the callback
            if self.callback:
                self.callback(self.labels[self.active])
                
            self.fig.canvas.draw_idle()
            
# Keep the SimpleDropdown class for backward compatibility
class SimpleDropdown(ScrollableDropdown):
    """
    Legacy simple dropdown class (redirects to ScrollableDropdown)
    """
    pass


class SynchronizedVideoPlayer:
    """
    A video player that synchronizes biomechanical data with video frames.
    """
    def __init__(
        self, video_path, insole_data, qtm_force_data, opencap_joint_data, qtm_joint_data,
        insole_offset=DEFAULT_INSOLE_OFFSET,
        qtm_force_offset=DEFAULT_QTM_FORCE_OFFSET,
        opencap_knee_offset=DEFAULT_OPENCAP_KNEE_OFFSET,
        qtm_knee_offset=DEFAULT_QTM_KNEE_OFFSET
    ):
        """
        Initialize the synchronized video player.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file
        insole_data : pandas.DataFrame
            Insole pressure data
        qtm_force_data : pandas.DataFrame
            QTM force plate data
        opencap_joint_data : pandas.DataFrame
            OpenCap joint angle data
        qtm_joint_data : pandas.DataFrame
            QTM joint angle data
        insole_offset : float, optional
            Time offset for insole data in seconds
        qtm_force_offset : float, optional
            Time offset for QTM force data in seconds
        opencap_knee_offset : float, optional
            Time offset for OpenCap joint data in seconds
        qtm_knee_offset : float, optional
            Time offset for QTM joint data in seconds
        """
        self.video_path = video_path
        self.insole_data = insole_data
        self.qtm_force_data = qtm_force_data
        self.opencap_joint_data = opencap_joint_data
        self.qtm_joint_data = qtm_joint_data

        # Initialize time offsets for each data stream (in seconds)
        self.insole_offset = insole_offset
        self.qtm_force_offset = qtm_force_offset
        self.opencap_joint_offset = opencap_knee_offset  # renamed but keeping params the same
        self.qtm_joint_offset = qtm_knee_offset  # renamed but keeping params the same

        # Track which kinematic parameters to plot
        self.selected_opencap_param = None
        self.selected_qtm_param = None
        
        # Get available parameters for dropdowns
        self.opencap_params = []
        if self.opencap_joint_data is not None:
            self.opencap_params = [col for col in self.opencap_joint_data.columns 
                                  if col != "time" and not col.startswith("Time")]
            if self.opencap_params:
                # Sort parameters for better dropdown organization
                self.opencap_params = sorted(self.opencap_params)
                self.selected_opencap_param = self.opencap_params[0]
                
        self.qtm_params = []
        if self.qtm_joint_data is not None:
            self.qtm_params = [col for col in self.qtm_joint_data.columns 
                              if col != "TimeSeconds" and not col.startswith("Time")]
            if self.qtm_params:
                # Sort parameters for better dropdown organization
                self.qtm_params = sorted(self.qtm_params)
                self.selected_qtm_param = self.qtm_params[0]

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
        if self.opencap_joint_data is not None:
            data_frames.append((self.opencap_joint_data, "time"))
        if self.qtm_joint_data is not None:
            data_frames.append((self.qtm_joint_data, "TimeSeconds"))

        self.sync_frames, self.common_time = synchronize_data(self.fps, *data_frames)

        # Set up UI
        self._setup_ui()
        
        # Set up keyboard listeners
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        
        # Animation
        self.is_playing = False
        self.anim = None

    def _setup_ui(self):
        """Set up the user interface with plots and controls"""
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

        # Create dropdown menu axes with fixed positioning
        opencap_panel_pos = self.gs_middle[2, 0].get_position(self.fig)
        qtm_panel_pos = self.gs_middle[3, 0].get_position(self.fig)
        
        # Create dropdown buttons and labels - wider dropdowns for better usability
        dropdown_width = 0.15
        dropdown_height = 0.04
        dropdown_left_pos = self.gs[0, 1].get_position(self.fig).x0 - 0.19  # Left position
        
        # Create the axes for dropdown menus
        self.ax_opencap_dropdown = self.fig.add_axes([
            dropdown_left_pos,  # Left
            opencap_panel_pos.y0 + (opencap_panel_pos.height / 2) - (dropdown_height / 2),  # Bottom
            dropdown_width,  # Width
            dropdown_height  # Height
        ])
        
        # Create label for OpenCap dropdown
        self.ax_opencap_label = self.fig.add_axes([
            dropdown_left_pos,  # Left
            opencap_panel_pos.y0 + opencap_panel_pos.height - 0.01,  # Top of panel
            dropdown_width,  # Width
            0.02  # Height
        ])
        
        # Create the axes for QTM dropdown
        self.ax_qtm_dropdown = self.fig.add_axes([
            dropdown_left_pos,  # Left
            qtm_panel_pos.y0 + (qtm_panel_pos.height / 2) - (dropdown_height / 2),  # Bottom
            dropdown_width,  # Width
            dropdown_height  # Height
        ])
        
        # Create label for QTM dropdown
        self.ax_qtm_label = self.fig.add_axes([
            dropdown_left_pos,  # Left
            qtm_panel_pos.y0 + qtm_panel_pos.height - 0.01,  # Top of panel
            dropdown_width,  # Width
            0.02  # Height
        ])

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

        # Set up the labels for the parameter selection dropdowns
        self.ax_opencap_label.set_axis_off()
        self.ax_opencap_label.text(0.5, 0.5, "OpenCap", 
                                  ha='center', va='center', fontsize=8, fontweight='bold')
        
        self.ax_qtm_label.set_axis_off()
        self.ax_qtm_label.text(0.5, 0.5, "QTM", 
                              ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Create scrollable dropdown menus
        if self.opencap_params:
            self.opencap_dropdown = ScrollableDropdown(
                self.ax_opencap_dropdown, 
                self.opencap_params, 
                active=0, 
                callback=self.on_opencap_param_change,
                max_display=8  # Show up to 8 items at once
            )
        
        if self.qtm_params:
            self.qtm_dropdown = ScrollableDropdown(
                self.ax_qtm_dropdown, 
                self.qtm_params, 
                active=0, 
                callback=self.on_qtm_param_change,
                max_display=8  # Show up to 8 items at once
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
        self._setup_offset_controls()

        # Set up the data plots
        self.setup_plots()

    def on_opencap_param_change(self, param):
        """Handle OpenCap parameter dropdown change"""
        self.selected_opencap_param = param
        self.update_plots()
        
    def on_qtm_param_change(self, param):
        """Handle QTM parameter dropdown change"""
        self.selected_qtm_param = param
        self.update_plots()

    def _setup_offset_controls(self):
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

        # OpenCap joint data offset controls
        self.ax_opencap_knee_offset.set_title("OpenCap Offset")
        self.opencap_knee_offset_slider = Slider(
            self.ax_opencap_knee_offset,
            "sec",
            -offset_range,
            offset_range,
            valinit=self.opencap_joint_offset,
            valfmt="%1.2f",
        )
        self.opencap_knee_offset_slider.on_changed(self.on_opencap_knee_offset_change)

        # Add text box for OpenCap joint offset
        text_bottom = self.ax_opencap_knee_offset.get_position().y0 - 0.03
        self.ax_opencap_knee_text = plt.axes(
            [text_left, text_bottom, text_width, text_height]
        )
        self.opencap_knee_offset_textbox = TextBox(
            self.ax_opencap_knee_text,
            "Value: ",
            initial=f"{self.opencap_joint_offset:.2f}",
        )
        self.opencap_knee_offset_textbox.on_submit(
            self.on_opencap_knee_offset_text_change
        )

        # QTM joint data offset controls
        self.ax_qtm_knee_offset.set_title("QTM Joint Offset")
        self.qtm_knee_offset_slider = Slider(
            self.ax_qtm_knee_offset,
            "sec",
            -offset_range,
            offset_range,
            valinit=self.qtm_joint_offset,
            valfmt="%1.2f",
        )
        self.qtm_knee_offset_slider.on_changed(self.on_qtm_knee_offset_change)

        # Add text box for QTM joint offset
        text_bottom = self.ax_qtm_knee_offset.get_position().y0 - 0.03
        self.ax_qtm_knee_text = plt.axes(
            [text_left, text_bottom, text_width, text_height]
        )
        self.qtm_knee_offset_textbox = TextBox(
            self.ax_qtm_knee_text, "Value: ", initial=f"{self.qtm_joint_offset:.2f}"
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
        """Handle OpenCap joint offset slider change"""
        self.opencap_joint_offset = val
        self.opencap_knee_offset_textbox.set_val(f"{val:.2f}")
        self.update_plots()

    def on_opencap_knee_offset_text_change(self, val):
        """Handle OpenCap joint offset text input change"""
        try:
            offset = float(val)
            self.opencap_joint_offset = offset
            self.opencap_knee_offset_slider.set_val(offset)
            self.update_plots()
        except ValueError:
            # Restore original value if input is invalid
            self.opencap_knee_offset_textbox.set_val(f"{self.opencap_joint_offset:.2f}")

    def on_qtm_knee_offset_change(self, val):
        """Handle QTM joint offset slider change"""
        self.qtm_joint_offset = val
        self.qtm_knee_offset_textbox.set_val(f"{val:.2f}")
        self.update_plots()

    def on_qtm_knee_offset_text_change(self, val):
        """Handle QTM joint offset text input change"""
        try:
            offset = float(val)
            self.qtm_joint_offset = offset
            self.qtm_knee_offset_slider.set_val(offset)
            self.update_plots()
        except ValueError:
            # Restore original value if input is invalid
            self.qtm_knee_offset_textbox.set_val(f"{self.qtm_joint_offset:.2f}")

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

        # OpenCap joint angles plot - only the selected parameter
        if self.opencap_joint_data is not None and self.selected_opencap_param:
            opencap_joint_sync = self.sync_frames[idx]
            idx += 1

            # Apply offset to time values
            adjusted_time = opencap_joint_sync["time"] + self.opencap_joint_offset

            # Plot only the selected parameter
            if self.selected_opencap_param in opencap_joint_sync.columns:
                # Apply filtering
                angles = filter_signal(opencap_joint_sync[self.selected_opencap_param].values)

                label = self.selected_opencap_param.replace("_", " ").title()
                self.ax_opencap_knee.plot(
                    adjusted_time, angles, label=label, linewidth=1.2, color='blue'
                )

                self.ax_opencap_knee.set_ylabel(f"OpenCap {label} (°)")
                self.ax_opencap_knee.legend(loc="upper right")
                self.ax_opencap_knee.grid(True)

            # Add cursor line
            cursor_line = self.ax_opencap_knee.axvline(
                x=0, color="k", linestyle="--", linewidth=0.8
            )
            self.cursor_lines.append(cursor_line)

        # QTM joint angles plot - only the selected parameter
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
                self.ax_qtm_knee.plot(
                    adjusted_time, angles, label=label, linewidth=1.2, color='red'
                )

                self.ax_qtm_knee.set_ylabel(f"QTM {label} (°)")
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

        # Don't use tight_layout as it's not compatible with our custom axes
        # plt.tight_layout()

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