"""
Window for ML dataset generation and visualization
"""

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QHeaderView,
    QSplitter,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QScrollArea,
    QFrame,
    QListWidget,
    QListWidgetItem
)
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

# Store reference to avoid garbage collection
_ml_model_window_ref = None

class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plot embedding in Qt"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)


class MLDatasetWindow(QMainWindow):
    """Window for displaying and exporting ML dataset"""
    
    def __init__(self, title, dataset, parent=None):
        super(MLDatasetWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)
        self.dataset = dataset
        
        # Initialize segment selection
        self.selected_segments = []
        
        # Verify and report dataset contents
        self._log_dataset_info()
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for visualization and controls
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Top part - Data visualization
        visualization_widget = QWidget()
        visualization_layout = QVBoxLayout(visualization_widget)
        
        # Create a horizontal layout for visualization controls and the chart
        viz_container = QWidget()
        viz_container_layout = QHBoxLayout(viz_container)
        viz_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chart visibility controls
        viz_control_widget = QWidget()
        viz_control_widget.setMaximumWidth(200)
        viz_control_layout = QVBoxLayout(viz_control_widget)
        
        # Add title for the controls
        viz_control_layout.addWidget(QLabel("<b>Chart Visibility</b>"))
        
        # Define data series categories for visibility control
        self.category_checkboxes = {}
        
        # Insole forces
        insole_check = QCheckBox("Insole Forces")
        insole_check.setChecked(True)
        insole_check.stateChanged.connect(self.update_chart_visibility)
        viz_control_layout.addWidget(insole_check)
        self.category_checkboxes['Insole Forces'] = insole_check
        
        # OpenCap kinematics
        opencap_check = QCheckBox("OpenCap Kinematics")
        opencap_check.setChecked(True)
        opencap_check.stateChanged.connect(self.update_chart_visibility)
        viz_control_layout.addWidget(opencap_check)
        self.category_checkboxes['OpenCap Kinematics'] = opencap_check
        
        # QTM Force
        qtm_force_check = QCheckBox("QTM Force")
        qtm_force_check.setChecked(True)
        qtm_force_check.stateChanged.connect(self.update_chart_visibility)
        viz_control_layout.addWidget(qtm_force_check)
        self.category_checkboxes['QTM Force'] = qtm_force_check
        
        # QTM Kinematics
        qtm_kine_check = QCheckBox("QTM Kinematics")
        qtm_kine_check.setChecked(True)
        qtm_kine_check.stateChanged.connect(self.update_chart_visibility)
        viz_control_layout.addWidget(qtm_kine_check)
        self.category_checkboxes['QTM Kinematics'] = qtm_kine_check
        
        # Joint Moments
        moments_check = QCheckBox("Joint Moments")
        moments_check.setChecked(True)
        moments_check.stateChanged.connect(self.update_chart_visibility)
        viz_control_layout.addWidget(moments_check)
        self.category_checkboxes['Joint Moments'] = moments_check
        
        # Select All / None
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_categories)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_no_categories)
        
        select_btns_layout = QHBoxLayout()
        select_btns_layout.addWidget(select_all_btn)
        select_btns_layout.addWidget(select_none_btn)
        viz_control_layout.addLayout(select_btns_layout)
        
        # Add segment selection controls
        segment_group = QGroupBox("Segment Selection")
        segment_layout = QVBoxLayout(segment_group)
        
        # Instructions label
        segment_layout.addWidget(QLabel("Drag on chart to select segments"))
        
        # Selected segments list
        self.segment_list = QListWidget()
        segment_layout.addWidget(self.segment_list)
        
        # Segment action buttons
        segment_buttons_layout = QHBoxLayout()
        
        self.clear_segments_btn = QPushButton("Clear All")
        self.clear_segments_btn.clicked.connect(self.clear_segments)
        segment_buttons_layout.addWidget(self.clear_segments_btn)
        
        self.remove_segment_btn = QPushButton("Remove Selected")
        self.remove_segment_btn.clicked.connect(self.remove_selected_segment)
        segment_buttons_layout.addWidget(self.remove_segment_btn)
        
        segment_layout.addLayout(segment_buttons_layout)
        
        # Export only selected segments checkbox
        self.export_selected_only = QCheckBox("Export only selected segments")
        self.export_selected_only.setChecked(True)
        segment_layout.addWidget(self.export_selected_only)
        
        # Add to control layout
        viz_control_layout.addWidget(segment_group)
        
        # Add stretch to push controls to the top
        viz_control_layout.addStretch()
        
        # Create matplotlib canvas for the plot
        self.canvas = MatplotlibCanvas(width=10, height=5, dpi=100)
        
        # Add controls and canvas to the viz container
        viz_container_layout.addWidget(viz_control_widget)
        viz_container_layout.addWidget(self.canvas, 1)  # Give the canvas more space
        
        # Add the viz container to the main visualization layout
        visualization_layout.addWidget(viz_container)
        
        # Bottom part - Dataset preview and export
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Dataset summary
        summary_group = QGroupBox("Dataset Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        # Create summary labels
        self.summary_data_points = QLabel(f"Data Points: {len(self.dataset)}")
        summary_layout.addWidget(self.summary_data_points)
        
        # Count different types of columns
        insole_cols = [col for col in self.dataset.columns if 'insole' in col.lower()]
        opencap_cols = [col for col in self.dataset.columns if 'opencap' in col.lower()]
        qtm_force_cols = [col for col in self.dataset.columns if 'force' in col.lower() and 'qtm' in col.lower()]
        qtm_kine_cols = [col for col in self.dataset.columns if 'qtm' in col.lower() 
                        and 'moment' not in col.lower() and 'force' not in col.lower()]
        moment_cols = [col for col in self.dataset.columns if 'moment' in col.lower()]
        
        self.summary_features = QLabel(f"Input Features: {len(insole_cols + opencap_cols + qtm_force_cols + qtm_kine_cols)}")
        summary_layout.addWidget(self.summary_features)
        
        self.summary_targets = QLabel(f"Target Variables: {len(moment_cols)}")
        summary_layout.addWidget(self.summary_targets)
        
        # Joint moments section
        if moment_cols:
            moments_layout = QVBoxLayout()
            moments_layout.addWidget(QLabel("Joint Moments (Target Variables):"))
            
            for col in moment_cols:
                moment_label = QLabel(f"• {col}")
                moments_layout.addWidget(moment_label)
            
            summary_layout.addLayout(moments_layout)
        else:
            warning_label = QLabel("⚠️ WARNING: No joint moments found in the dataset!")
            warning_label.setStyleSheet("color: red; font-weight: bold;")
            summary_layout.addWidget(warning_label)
        
        control_layout.addWidget(summary_group)
        
        # Dataset preview
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Create a table widget for dataset preview
        self.table_widget = QTableWidget()
        preview_layout.addWidget(self.table_widget)
        
        control_layout.addWidget(preview_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout(export_group)
        
        # Button for exporting to CSV
        self.export_csv_button = QPushButton("Export to CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        export_layout.addWidget(self.export_csv_button)
        
        # Button for exporting to Excel
        self.export_excel_button = QPushButton("Export to Excel")
        self.export_excel_button.clicked.connect(self.export_to_excel)
        export_layout.addWidget(self.export_excel_button)
        
        # Button for exporting to NumPy format
        self.export_numpy_button = QPushButton("Export to NumPy")
        self.export_numpy_button.clicked.connect(self.export_to_numpy)
        export_layout.addWidget(self.export_numpy_button)
        
        # Button for launching ML model training
        self.train_ml_button = QPushButton("Train ML Model")
        self.train_ml_button.clicked.connect(self.launch_model_training)
        export_layout.addWidget(self.train_ml_button)
        
        control_layout.addWidget(export_group)
        
        # Add widgets to splitter
        splitter.addWidget(visualization_widget)
        splitter.addWidget(control_widget)
        
        # Set splitter sizes
        splitter.setSizes([600, 200])
        
        # Initialize the UI with data
        self.load_dataset()
        
    def _log_dataset_info(self):
        """Log information about the dataset for debugging"""
        if self.dataset is None:
            print("Dataset is None")
            return
            
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"Dataset columns: {list(self.dataset.columns)}")
        
        # Check for specific types of columns
        insole_cols = [col for col in self.dataset.columns if 'insole' in col.lower()]
        opencap_cols = [col for col in self.dataset.columns if 'opencap' in col.lower()]
        qtm_force_cols = [col for col in self.dataset.columns if 'force' in col.lower() and 'qtm' in col.lower()]
        qtm_kine_cols = [col for col in self.dataset.columns if 'qtm' in col.lower() 
                        and 'moment' not in col.lower() and 'force' not in col.lower()]
        moment_cols = [col for col in self.dataset.columns if 'moment' in col.lower()]
        
        print(f"Found {len(insole_cols)} insole columns: {insole_cols}")
        print(f"Found {len(opencap_cols)} OpenCap columns: {opencap_cols}")
        print(f"Found {len(qtm_force_cols)} QTM force columns: {qtm_force_cols}")
        print(f"Found {len(qtm_kine_cols)} QTM kinematics columns: {qtm_kine_cols}")
        print(f"Found {len(moment_cols)} moment columns: {moment_cols}")
        
    def load_dataset(self):
        """Load and display the dataset"""
        if self.dataset is None or len(self.dataset) == 0:
            return
            
        # Update table with dataset preview
        self.update_table()
        
        # Plot the data
        self.plot_data()
    
    def update_table(self):
        """Update the table widget with dataset preview"""
        # Use pandas DataFrame for convenience
        df = self.get_filtered_dataset() if self.export_selected_only.isChecked() else self.dataset
        
        # Set table dimensions
        self.table_widget.setRowCount(min(10, len(df)))  # Show first 10 rows
        self.table_widget.setColumnCount(len(df.columns))
        
        # Set headers
        self.table_widget.setHorizontalHeaderLabels(df.columns)
        
        # Fill table with data
        for i in range(min(10, len(df))):
            for j in range(len(df.columns)):
                value = str(df.iloc[i, j])
                item = QTableWidgetItem(value)
                self.table_widget.setItem(i, j, item)
        
        # Resize columns to content
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Update summary
        self.summary_data_points.setText(f"Data Points: {len(df)}")
    
    def plot_data(self):
        """Plot the dataset for visualization"""
        # Create visibility dict with all categories set to True (default)
        visibility = {category: True for category in self.category_checkboxes.keys()}
        self.plot_data_with_visibility(visibility)
    
    def plot_data_with_visibility(self, visibility):
        """Plot the dataset for visualization with specific category visibility"""
        self.canvas.axes.clear()
        
        # Get time column
        time_col = self.dataset.columns[0]  # Assuming first column is time
        time_data = self.dataset[time_col]
        
        # Define different types of data for plotting with different colors
        data_categories = {
            'Insole Forces': {
                'pattern': 'insole',
                'color': 'green', 
                'alpha': 0.6,
                'linewidth': 0.8,
                'visible': visibility['Insole Forces']
            },
            'OpenCap Kinematics': {
                'pattern': 'opencap_',
                'color': 'blue', 
                'alpha': 0.6,
                'linewidth': 0.8,
                'visible': visibility['OpenCap Kinematics']
            },
            'QTM Force': {
                'pattern': 'qtm_force',
                'color': 'magenta',
                'alpha': 0.6,
                'linewidth': 0.8,
                'visible': visibility['QTM Force']
            },
            'QTM Kinematics': {
                'pattern': 'qtm_',
                'exclude_pattern': 'moment',
                'exclude_pattern2': 'force',
                'color': 'orange',
                'alpha': 0.6,
                'linewidth': 0.8,
                'visible': visibility['QTM Kinematics']
            },
            'Joint Moments': {
                'pattern': 'moment',
                'color': 'red',
                'alpha': 1.0,
                'linewidth': 1.2,
                'visible': visibility['Joint Moments']
            }
        }
        
        # Only count visible categories for calculating the maximum series per category
        visible_categories = sum(1 for cat, config in data_categories.items() if config['visible'])
        max_series_per_category = max(1, int(15 / max(1, visible_categories)))
        
        # Plot each category with its specific style
        legend_handles = []
        legend_labels = []
        
        for category, config in data_categories.items():
            # Skip if the category is not visible
            if not config['visible']:
                continue
                
            # Get columns matching this category
            matching_cols = []
            for col in self.dataset.columns:
                if config['pattern'] in col.lower():
                    # Check for exclusion patterns if they exist
                    if 'exclude_pattern' in config and config['exclude_pattern'] in col.lower():
                        continue
                    if 'exclude_pattern2' in config and config['exclude_pattern2'] in col.lower():
                        continue
                    matching_cols.append(col)
            
            # Sort columns alphabetically for consistent display
            matching_cols.sort()
            
            # Limit to max number of series per category
            if len(matching_cols) > max_series_per_category:
                # Choose representative columns if we have too many
                step = len(matching_cols) // max_series_per_category
                matching_cols = matching_cols[::step][:max_series_per_category]
            
            # Plot each column
            for col in matching_cols:
                line, = self.canvas.axes.plot(
                    time_data, 
                    self.dataset[col], 
                    color=config['color'],
                    alpha=config['alpha'],
                    linewidth=config['linewidth'],
                    label=col
                )
                
                # Only add to legend for the first item in each category to avoid overcrowding
                if col == matching_cols[0]:
                    legend_handles.append(line)
                    legend_labels.append(category)
        
        # Add legend with category-based grouping, but only if there are visible series
        if legend_handles:
            self.canvas.axes.legend(
                legend_handles,
                legend_labels,
                loc='upper right', 
                fontsize='small'
            )
        
        # Set labels
        self.canvas.axes.set_xlabel('Time (s)')
        self.canvas.axes.set_ylabel('Value')
        self.canvas.axes.set_title('ML Dataset Visualization')
        self.canvas.axes.grid(True)
        
        # Draw selected segments
        self._draw_segments()
        
        # Add span selector for selecting segments
        self._add_span_selector()
        
        # Tight layout and draw
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def _add_span_selector(self):
        """Add a span selector to the plot for selecting time segments"""
        self.span_selector = SpanSelector(
            self.canvas.axes,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='blue'),
            interactive=True,
            drag_from_anywhere=True
        )
    
    def on_select(self, xmin, xmax):
        """Handle selection of a time segment"""
        if xmin == xmax:
            return  # Avoid zero-width selections
        
        # Round to 2 decimal places for clarity
        xmin = round(xmin, 2)
        xmax = round(xmax, 2)
        
        # Add to selected segments if not already there
        segment = (xmin, xmax)
        if segment not in self.selected_segments:
            self.selected_segments.append(segment)
            self._update_segment_list()
            self._draw_segments()
            
            # Update the table preview if we're showing filtered data
            if self.export_selected_only.isChecked():
                self.update_table()
    
    def _update_segment_list(self):
        """Update the list widget with selected segments"""
        self.segment_list.clear()
        
        # Sort segments by start time
        self.selected_segments.sort(key=lambda x: x[0])
        
        for i, (start, end) in enumerate(self.selected_segments):
            self.segment_list.addItem(f"Segment {i+1}: {start:.2f}s - {end:.2f}s")
    
    def _draw_segments(self):
        """Draw the selected segments on the plot"""
        # Get current y limits
        y_min, y_max = self.canvas.axes.get_ylim()
        
        # Remove existing segment rectangles 
        for artist in self.canvas.axes.get_children():
            if hasattr(artist, '_is_segment_rect'):
                artist.remove()
        
        # Draw new rectangles for each segment
        for start, end in self.selected_segments:
            rect = plt.Rectangle(
                (start, y_min),
                end - start,
                y_max - y_min,
                facecolor='blue',
                alpha=0.2,
                edgecolor='blue',
                linewidth=1
            )
            rect._is_segment_rect = True
            self.canvas.axes.add_patch(rect)
        
        self.canvas.draw()
    
    def clear_segments(self):
        """Clear all selected segments"""
        self.selected_segments = []
        self._update_segment_list()
        self._draw_segments()
        
        # Update the table preview if we're showing filtered data
        if self.export_selected_only.isChecked():
            self.update_table()
    
    def remove_selected_segment(self):
        """Remove the currently selected segment in the list"""
        selected_items = self.segment_list.selectedItems()
        if not selected_items:
            return
            
        # Get the index from the item text (e.g., "Segment 1: 0.00s - 1.00s")
        item_text = selected_items[0].text()
        if item_text.startswith("Segment "):
            try:
                index = int(item_text.split(":")[0].replace("Segment ", "")) - 1
                if 0 <= index < len(self.selected_segments):
                    self.selected_segments.pop(index)
                    self._update_segment_list()
                    self._draw_segments()
                    
                    # Update the table preview if we're showing filtered data
                    if self.export_selected_only.isChecked():
                        self.update_table()
            except (ValueError, IndexError):
                pass
    
    def get_filtered_dataset(self):
        """Get dataset filtered by selected segments"""
        if not self.selected_segments or not self.export_selected_only.isChecked():
            return self.dataset
        
        # Get time column
        time_col = self.dataset.columns[0]  # Assuming first column is time
        time_data = self.dataset[time_col]
        
        # Create mask for all selected segments
        mask = np.zeros(len(time_data), dtype=bool)
        
        for start, end in self.selected_segments:
            segment_mask = (time_data >= start) & (time_data <= end)
            mask = mask | segment_mask
        
        # Return filtered dataset
        filtered_df = self.dataset[mask].copy()
        
        # Update the summary to show how many points are selected
        print(f"Selected {len(filtered_df)} out of {len(self.dataset)} data points")
        
        return filtered_df
    
    def export_to_csv(self):
        """Export dataset to CSV file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Dataset to CSV', '', 'CSV Files (*.csv)'
        )
        
        if file_path:
            try:
                # Use filtered dataset if option is checked
                dataset_to_export = self.get_filtered_dataset()
                
                dataset_to_export.to_csv(file_path, index=False)
                self.show_status_message(f"Dataset exported to {file_path}")
            except Exception as e:
                self.show_status_message(f"Error exporting to CSV: {str(e)}")
    
    def export_to_excel(self):
        """Export dataset to Excel file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Dataset to Excel', '', 'Excel Files (*.xlsx)'
        )
        
        if file_path:
            try:
                # Use filtered dataset if option is checked
                dataset_to_export = self.get_filtered_dataset()
                
                dataset_to_export.to_excel(file_path, index=False)
                self.show_status_message(f"Dataset exported to {file_path}")
            except Exception as e:
                self.show_status_message(f"Error exporting to Excel: {str(e)}")
    
    def export_to_numpy(self):
        """Export dataset to NumPy format"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Dataset to NumPy', '', 'NumPy Files (*.npz)'
        )
        
        if file_path:
            try:
                # Use filtered dataset if option is checked
                dataset_to_export = self.get_filtered_dataset()
                
                # Split into different variable types
                time_col = dataset_to_export.columns[0]  # Assuming first column is time
                
                # Identify different types of input features
                insole_cols = [col for col in dataset_to_export.columns 
                              if 'insole' in col.lower()]
                
                opencap_cols = [col for col in dataset_to_export.columns 
                               if 'opencap' in col.lower()]
                
                qtm_force_cols = [col for col in dataset_to_export.columns 
                                 if 'qtm_force' in col.lower()]
                
                qtm_kinematics_cols = [col for col in dataset_to_export.columns 
                                      if 'qtm_' in col.lower() 
                                      and 'moment' not in col.lower()
                                      and 'force' not in col.lower()]
                
                # Target variables (joint moments)
                target_cols = [col for col in dataset_to_export.columns 
                              if 'moment' in col.lower()]
                
                # Extract data arrays
                time_data = dataset_to_export[time_col].values
                
                # Organize by input type
                insole_data = dataset_to_export[insole_cols].values if insole_cols else np.array([])
                opencap_data = dataset_to_export[opencap_cols].values if opencap_cols else np.array([])
                qtm_force_data = dataset_to_export[qtm_force_cols].values if qtm_force_cols else np.array([])
                qtm_kinematics_data = dataset_to_export[qtm_kinematics_cols].values if qtm_kinematics_cols else np.array([])
                
                # All combined inputs for convenience
                all_input_cols = insole_cols + opencap_cols + qtm_force_cols + qtm_kinematics_cols
                all_inputs = dataset_to_export[all_input_cols].values if all_input_cols else np.array([])
                
                # Target outputs
                targets = dataset_to_export[target_cols].values if target_cols else np.array([])
                
                # Save to numpy format with explicit categorization
                np.savez(
                    file_path,
                    # Time
                    time=time_data,
                    
                    # Input features by category
                    insole_data=insole_data,
                    insole_cols=np.array(insole_cols),
                    
                    opencap_data=opencap_data,
                    opencap_cols=np.array(opencap_cols),
                    
                    qtm_force_data=qtm_force_data,
                    qtm_force_cols=np.array(qtm_force_cols),
                    
                    qtm_kinematics_data=qtm_kinematics_data,
                    qtm_kinematics_cols=np.array(qtm_kinematics_cols),
                    
                    # Combined inputs for convenience
                    inputs=all_inputs,
                    input_cols=np.array(all_input_cols),
                    
                    # Target variables
                    targets=targets,
                    target_cols=np.array(target_cols)
                )
                
                self.show_status_message(f"Dataset exported to {file_path}")
                print(f"Exported dataset with shapes: inputs={all_inputs.shape}, targets={targets.shape}")
                
            except Exception as e:
                self.show_status_message(f"Error exporting to NumPy: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def launch_model_training(self):
        """Launch the ML model training window"""
        print("Launching model training window")
        
        # Disable button to prevent multiple clicks
        self.train_ml_button.setEnabled(False)
        self.train_ml_button.setText("Loading...")
        
        try:
            # Import here to avoid circular imports
            from .ml_dataset_module import show_ml_model_window
            
            # Get the prepared dataset
            dataset = self.get_filtered_dataset()
            if dataset is None or len(dataset) < 10:
                print("Warning: Dataset is empty or too small")
                self.train_ml_button.setEnabled(True)
                self.train_ml_button.setText("Train ML Model")
                return
                
            # Process events to keep UI responsive
            from PyQt5.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            
            # Call the window creation function
            window = show_ml_model_window(self, dataset)
            
            # Store window reference to prevent garbage collection
            if window is not None:
                self.model_window = window
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error launching model training: {str(e)}")
            
            # Reset button state
            self.train_ml_button.setEnabled(True)
            self.train_ml_button.setText("Train ML Model")
            
            # Show error message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to launch model training window:\n{str(e)}"
            )
    
    def show_status_message(self, message):
        """Show a status message (in the future, add a status bar)"""
        print(message)  # For now, just print to console 

    def update_chart_visibility(self):
        """Update chart visibility based on checkbox states"""
        # Get the checked state of each category
        visibility = {
            'Insole Forces': self.category_checkboxes['Insole Forces'].isChecked(),
            'OpenCap Kinematics': self.category_checkboxes['OpenCap Kinematics'].isChecked(),
            'QTM Force': self.category_checkboxes['QTM Force'].isChecked(),
            'QTM Kinematics': self.category_checkboxes['QTM Kinematics'].isChecked(),
            'Joint Moments': self.category_checkboxes['Joint Moments'].isChecked()
        }
        
        # If we have an existing plot, update it
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.plot_data_with_visibility(visibility)
    
    def select_all_categories(self):
        """Select all categories"""
        for checkbox in self.category_checkboxes.values():
            checkbox.setChecked(True)
    
    def select_no_categories(self):
        """Select no categories"""
        for checkbox in self.category_checkboxes.values():
            checkbox.setChecked(False) 