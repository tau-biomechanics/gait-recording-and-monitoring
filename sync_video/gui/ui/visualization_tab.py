"""
Visualization tab component for the ML model window
"""

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
    )
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import sys
import traceback


class VisualizationTab:
    """
    UI Component for ML model visualization tab
    """
    
    def __init__(self, parent):
        """
        Initialize the visualization tab
        
        Parameters
        ----------
        parent : MLModelWindow
            Parent window containing the tab
        """
        self.parent = parent
        
    def setup_ui(self):
        """Set up the visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        try:
            print("========== SETTING UP VISUALIZATION TAB - START ==========")
            sys.stdout.flush()

            # Add a message
            layout.addWidget(
                QLabel("Visualization features will be available after model training")
            )
            
            # Create prediction canvas for visualizing model predictions
            from ...ml import ModelVisualizer
            self.parent.prediction_canvas = ModelVisualizer.create_prediction_canvas(self.parent)
            layout.addWidget(self.parent.prediction_canvas)

            # Add visualization button (disabled for now)
            self.parent.visualize_btn = QPushButton("Visualize Predictions")
            self.parent.visualize_btn.clicked.connect(self.parent.visualize_predictions)
            self.parent.visualize_btn.setEnabled(False)
            layout.addWidget(self.parent.visualize_btn)
            
            # Add model architecture visualization button
            self.parent.visualize_architecture_btn = QPushButton("Visualize Model Architecture")
            self.parent.visualize_architecture_btn.clicked.connect(self.parent.visualize_model_architecture)
            self.parent.visualize_architecture_btn.setEnabled(False)
            layout.addWidget(self.parent.visualize_architecture_btn)

            # Add export button (disabled for now)
            self.parent.export_btn = QPushButton("Export Predictions")
            self.parent.export_btn.clicked.connect(self.parent.export_predictions)
            self.parent.export_btn.setEnabled(False)
            layout.addWidget(self.parent.export_btn)

            # Add save/load model buttons
            button_layout = QHBoxLayout()

            self.parent.save_model_btn = QPushButton("Save Model")
            self.parent.save_model_btn.clicked.connect(self.parent.save_model)
            self.parent.save_model_btn.setEnabled(False)
            button_layout.addWidget(self.parent.save_model_btn)

            self.parent.load_model_btn = QPushButton("Load Model")
            self.parent.load_model_btn.clicked.connect(self.parent.load_model)
            button_layout.addWidget(self.parent.load_model_btn)

            layout.addLayout(button_layout)

            print("========== VISUALIZATION TAB SETUP COMPLETE ==========")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"ERROR in setting up visualization tab: {str(e)}")
            sys.stdout.flush()
            traceback.print_exc()
            # Keep a message about the error
            error_label = QLabel(f"Error setting up visualization tab: {str(e)}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)
            
        return tab