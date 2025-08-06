"""
Correlation analysis window for comparing time series data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLabel, 
    QTextEdit,
    QSplitter,
    QFrame,
    QPushButton,
    QFileDialog,
    QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import os


class CorrelationPlotWindow(QMainWindow):
    """
    Window for displaying correlation analysis between two data series
    """
    
    def __init__(self, title, metrics, series1_name, series2_name, parent=None):
        """
        Initialize the correlation window
        
        Parameters:
        -----------
        title : str
            Window title
        metrics : dict
            Dictionary of calculated correlation metrics
        series1_name : str
            Name of the first data series
        series2_name : str
            Name of the second data series
        parent : QWidget, optional
            Parent widget
        """
        super(CorrelationPlotWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)
        
        # Store parameters
        self.metrics = metrics
        self.series1_name = series1_name
        self.series2_name = series2_name
        
        # Set up the UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface"""
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for plots and metrics
        splitter = QSplitter(Qt.Vertical)
        
        # Upper part - plots
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure for plots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        
        # Create a grid of subplots
        self.gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1])
        
        # Time series comparison plot
        self.ax1 = self.fig.add_subplot(self.gs[0, :])
        
        # Cross-correlation plot
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])
        
        # Scatter plot
        self.ax3 = self.fig.add_subplot(self.gs[1, 1])
        
        # Create canvas
        self.canvas = FigureCanvasQTAgg(self.fig)
        plots_layout.addWidget(self.canvas)
        
        # Lower part - metrics and buttons
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Header layout with label and save button
        header_layout = QHBoxLayout()
        
        # Header label
        header_label = QLabel("Correlation Analysis Metrics")
        header_label.setAlignment(Qt.AlignCenter)
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(14)
        header_label.setFont(header_font)
        header_layout.addWidget(header_label)
        
        # Add save button
        self.save_button = QPushButton("Save as Image/PDF")
        self.save_button.setToolTip("Save this analysis as an image or PDF file")
        self.save_button.clicked.connect(self.save_analysis)
        header_layout.addWidget(self.save_button)
        
        # Add header layout to metrics layout
        metrics_layout.addLayout(header_layout)
        
        # Metrics text area
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        
        # Add widgets to splitter
        splitter.addWidget(plots_widget)
        splitter.addWidget(metrics_widget)
        
        # Set initial sizes (70% plots, 30% metrics)
        splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Plot the data
        self.plot_correlation()
        
    def save_analysis(self):
        """Save the analysis as an image or PDF file"""
        try:
            # Create file dialog
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Save Analysis")
            
            # Set default filename based on series names
            default_filename = f"correlation_{self.series1_name.replace(' ', '_')}_vs_{self.series2_name.replace(' ', '_')}"
            
            # Set filter for file types
            file_dialog.setNameFilter("Images (*.png *.jpg);;PDF Files (*.pdf);;All Files (*)")
            file_dialog.selectNameFilter("Images (*.png *.jpg)")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.setDirectory(os.path.expanduser("~"))
            file_dialog.selectFile(default_filename)
            
            # Execute dialog
            if file_dialog.exec_() == QFileDialog.Accepted:
                # Get selected file path
                file_path = file_dialog.selectedFiles()[0]
                
                # Ensure file has an extension
                if not any(file_path.endswith(ext) for ext in ['.png', '.jpg', '.pdf']):
                    # Add default extension based on selected filter
                    selected_filter = file_dialog.selectedNameFilter()
                    if "Images" in selected_filter:
                        file_path += '.png'
                    elif "PDF" in selected_filter:
                        file_path += '.pdf'
                
                # Simply save the current figure as displayed in the window
                self.fig.savefig(file_path, bbox_inches='tight', dpi=300)
                
                # Show success message
                QMessageBox.information(self, "Save Successful", 
                                    f"Plot saved to:\n{file_path}")
        
        except Exception as e:
            # Show error message
            QMessageBox.critical(self, "Save Error", 
                              f"An error occurred while saving:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
    def plot_correlation(self):
        """Plot correlation data and update metrics text"""
        if 'error' in self.metrics:
            # Display error message
            self.metrics_text.setText(f"Error: {self.metrics['error']}")
            return
        
        # Extract data from metrics
        time = self.metrics['time_values']
        series1 = self.metrics['series1_resampled']
        series2 = self.metrics['series2_resampled']
        lags = self.metrics['lags']
        cross_corr = self.metrics['cross_correlation']
        
        # Plot time series comparison
        self.ax1.clear()
        self.ax1.plot(time, series1, 'b-', label=self.series1_name, linewidth=1.5)
        self.ax1.plot(time, series2, 'r-', label=self.series2_name, linewidth=1.5)
        self.ax1.set_title('Time Series Comparison')
        self.ax1.set_xlabel('Time (s)', fontsize=12)
        self.ax1.set_ylabel('Value', fontsize=12)
        self.ax1.legend(loc='upper right', fontsize=11)
        self.ax1.grid(True)
        self.ax1.tick_params(labelsize=10)
        
        # Add correlation coefficient annotation with better visibility
        pearson_r = self.metrics['pearson_r']
        # Calculate correct R² (should always be positive)
        r_squared = pearson_r * pearson_r  # Use proper R² calculation
        annotation = f"Pearson r = {pearson_r:.4f}\nR² = {r_squared:.4f}"
        
        # Use light background with border for better readability
        self.ax1.annotate(annotation, xy=(0.02, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
                         fontsize=11, fontweight='bold')
        
        # Plot cross-correlation
        self.ax2.clear()
        self.ax2.plot(lags, cross_corr, 'g-')
        self.ax2.set_title('Cross-Correlation')
        self.ax2.set_xlabel('Lag (s)', fontsize=12)
        self.ax2.set_ylabel('Correlation', fontsize=12)
        self.ax2.grid(True)
        self.ax2.tick_params(labelsize=10)
        
        # Mark the maximum correlation with improved visibility
        max_idx = np.argmax(np.abs(cross_corr))
        max_lag = lags[max_idx]
        max_corr = cross_corr[max_idx]
        self.ax2.plot(max_lag, max_corr, 'ro', markersize=8)
        
        # Create a separate annotation in the upper corner for better visibility
        max_annotation = f"Max correlation: {max_corr:.3f}\nOptimal lag: {max_lag:.3f}s"
        self.ax2.annotate(max_annotation, xy=(0.02, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
                         fontsize=11, fontweight='bold')
        
        # Plot scatter plot with regression line
        self.ax3.clear()
        self.ax3.scatter(series1, series2, alpha=0.5, s=12)
        self.ax3.set_title('Correlation Scatter Plot')
        self.ax3.set_xlabel(self.series1_name, fontsize=12)
        self.ax3.set_ylabel(self.series2_name, fontsize=12)
        self.ax3.grid(True)
        self.ax3.tick_params(labelsize=10)
        
        # Add regression line
        if len(series1) > 1:
            m, b = np.polyfit(series1, series2, 1)
            x_range = np.linspace(min(series1), max(series1), 100)
            self.ax3.plot(x_range, m * x_range + b, 'r-', linewidth=2)
            
            # Add regression equation with improved visibility
            eq_text = f"y = {m:.4f}x + {b:.4f}"
            self.ax3.annotate(eq_text, xy=(0.02, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
                             fontsize=11, fontweight='bold')
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update metrics text
        from ..processors.correlation_analysis import create_correlation_summary
        summary_text = create_correlation_summary(self.metrics, self.series1_name, self.series2_name)
        self.metrics_text.setText(summary_text)
        
        # Format the metrics text without setting background color to respect dark mode
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                font-family: Arial, sans-serif;
                font-size: 12pt;
                line-height: 1.5;
                padding: 10px;
            }
        """) 