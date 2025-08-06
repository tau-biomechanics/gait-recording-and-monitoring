"""
Window for training and visualizing CNN models for joint moment prediction
"""

try:
    from PyQt5.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QLabel,
        QTabWidget,
        QMessageBox,
    )
    from PyQt5.QtCore import QCoreApplication
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import numpy as np
import sys
import traceback
import time

from .utils import debug_log
from .ui import SimpleLoadingWindow, TrainingTab, EvaluationTab, VisualizationTab
from .threads import DataAnalyzeThread
from .model_manager import ModelManager
from .visualization_manager import VisualizationManager

print(f"ML Model Window module imported at {time.strftime('%H:%M:%S')}")


class MLModelWindow(QMainWindow):
    """Window for training and visualizing CNN models for joint moment prediction"""

    def __init__(self, dataset, parent=None):
        print("█████████████████████████████████████████")
        print("█ INITIALIZING ML MODEL WINDOW           █")
        print("█ Starting MLModelWindow initialization  █")
        print(f"█ Got dataset with shape: {dataset.shape} █")
        print("█████████████████████████████████████████")
        sys.stdout.flush()

        debug_log("Starting MLModelWindow initialization")

        # Start with simple loading window
        self.loading_window = SimpleLoadingWindow(dataset, parent)
        self.loading_window.show()

        # Process events to make sure loading window is displayed
        QCoreApplication.processEvents()

        # Print debug info
        debug_log(f"Creating ML Model Window: dataset shape {dataset.shape}")

        # Initialize the real window in the background
        super(MLModelWindow, self).__init__(parent)
        self.setWindowTitle("Joint Moment Prediction Model")
        self.setMinimumSize(1000, 700)
        self.dataset = dataset
        self.model = None
        self.training_thread = None
        self.training_data = None
        self.analysis_thread = None

        # Initialize managers
        self.model_manager = ModelManager(self)
        self.visualization_manager = VisualizationManager(self)

        # Store results from dataset analysis
        self.insole_cols = []
        self.opencap_cols = []
        self.qtm_force_cols = []
        self.qtm_kine_cols = []
        self.moment_cols = []
        self.time_col = None

        # Set up the main UI structure
        debug_log("Creating main UI")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tabs widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Status bar at the bottom
        self.status_label = QLabel("Initializing...")
        main_layout.addWidget(self.status_label)

        # Now start loading everything in stages
        debug_log("Creating tabs")
        self._create_tabs()

        # Start analysis
        debug_log("Starting dataset analysis")
        QCoreApplication.processEvents()
        self._run_analysis()
        debug_log("MLModelWindow initialization completed")

    def _create_tabs(self):
        """Create the basic structure of all tabs"""
        debug_log("Creating tab structure")
        self.loading_window.update_progress(10, "Setting up UI...", "Creating tabs")

        # Create tab components
        self.training_tab = TrainingTab(self)
        self.evaluation_tab = EvaluationTab(self)
        self.visualization_tab = VisualizationTab(self)

        # Add tabs with temporary loading message
        train_tab = QWidget()
        self.training_layout = QVBoxLayout(train_tab)
        self.training_layout.addWidget(QLabel("Loading..."))
        self.tabs.addTab(train_tab, "Training")

        eval_tab = QWidget()
        self.evaluation_layout = QVBoxLayout(eval_tab)
        self.evaluation_layout.addWidget(QLabel("Loading..."))
        self.tabs.addTab(eval_tab, "Evaluation")

        viz_tab = QWidget()
        self.viz_layout = QVBoxLayout(viz_tab)
        self.viz_layout.addWidget(QLabel("Loading..."))
        self.tabs.addTab(viz_tab, "Visualization")

        debug_log("Tab structure created")

    def _run_analysis(self):
        """Start the dataset analysis process"""
        debug_log("Starting dataset analysis process")
        self.loading_window.update_progress(
            20, "Analyzing dataset...", "Starting analysis thread"
        )

        # Create analysis thread
        try:
            debug_log("Creating analysis thread")
            self.analysis_thread = DataAnalyzeThread(self.dataset)

            # Connect signals
            debug_log("Connecting analysis thread signals")
            self.analysis_thread.progress_signal.connect(self._on_analysis_progress)
            self.analysis_thread.finished_signal.connect(self._on_analysis_complete)
            self.analysis_thread.error_signal.connect(self._on_analysis_error)

            # Start thread - this is a key point, after this the thread will run
            debug_log("*** STARTING ANALYSIS THREAD ***")
            print("Starting analysis thread with dataset shape:", self.dataset.shape)
            sys.stdout.flush()
            self.analysis_thread.start()
            debug_log("Analysis thread started")
        except Exception as e:
            print(f"ERROR STARTING ANALYSIS THREAD: {str(e)}")
            sys.stdout.flush()
            traceback.print_exc()
            self._on_analysis_error(f"Failed to start analysis: {str(e)}")

    def _on_analysis_progress(self, value, message, detail):
        """Handle analysis progress updates"""
        debug_log(f"Analysis progress: {value}% - {message}")
        self.loading_window.update_progress(20 + int(value * 0.4), message, detail)
        self.status_label.setText(message)
        # Force UI update
        QCoreApplication.processEvents()

    def _on_analysis_error(self, error_message):
        """Handle analysis errors"""
        debug_log(f"Analysis error: {error_message}")
        self.loading_window.update_progress(100, "Error", error_message)

        # Show error message in the loading window
        QMessageBox.critical(
            self.loading_window,
            "Analysis Error",
            f"Error during dataset analysis:\n{error_message}",
        )

        # Close the loading window
        self.loading_window.close()
        debug_log("Loading window closed due to error")

    def _on_analysis_complete(self, results):
        """Handle completed analysis"""
        debug_log("Analysis complete, results received")
        print("**** ANALYSIS COMPLETE CALLBACK STARTED ****")
        print(f"Got results with {len(results)} items")
        sys.stdout.flush()

        self.loading_window.update_progress(60, "Building training UI...", "")

        # Store analysis results
        debug_log("Storing analysis results")
        self.time_col = results.get("time_col", None)
        self.insole_cols = results.get("insole_cols", [])
        self.opencap_cols = results.get("opencap_cols", [])
        self.qtm_force_cols = results.get("qtm_force_cols", [])
        self.qtm_kine_cols = results.get("qtm_kine_cols", [])
        self.moment_cols = results.get("moment_cols", [])

        # Print what we found for debugging
        print(f"Time column: {self.time_col}")
        print(f"Found {len(self.insole_cols)} insole columns")
        print(f"Found {len(self.opencap_cols)} OpenCap columns")
        print(f"Found {len(self.qtm_force_cols)} QTM force columns")
        print(f"Found {len(self.qtm_kine_cols)} QTM kinematic columns")
        print(f"Found {len(self.moment_cols)} moment columns: {self.moment_cols}")
        sys.stdout.flush()

        # Complete UI setup in steps
        try:
            print("Starting UI setup steps")
            sys.stdout.flush()

            # First remove the loading message
            debug_log("Removing loading messages")
            for layout in [
                self.training_layout,
                self.evaluation_layout,
                self.viz_layout,
            ]:
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

            # Set up the training tab
            debug_log("Setting up training tab")
            print("Setting up training tab")
            sys.stdout.flush()
            self.loading_window.update_progress(
                65, "Building training UI...", "Setting up input controls"
            )
            training_widget = self.training_tab.setup_ui()
            self.tabs.removeTab(0)
            self.tabs.insertTab(0, training_widget, "Training")

            # Set up the evaluation tab
            debug_log("Setting up evaluation tab")
            print("Setting up evaluation tab")
            sys.stdout.flush()
            self.loading_window.update_progress(75, "Building evaluation UI...", "")
            evaluation_widget = self.evaluation_tab.setup_ui()
            self.tabs.removeTab(1)
            self.tabs.insertTab(1, evaluation_widget, "Evaluation")

            # Set up the visualization tab
            debug_log("Setting up visualization tab")
            print("Setting up visualization tab")
            sys.stdout.flush()
            self.loading_window.update_progress(85, "Building visualization UI...", "")
            visualization_widget = self.visualization_tab.setup_ui()
            self.tabs.removeTab(2)
            self.tabs.insertTab(2, visualization_widget, "Visualization")

            # Update status bar
            debug_log("Updating status bar")
            print("Updating status bar")
            sys.stdout.flush()
            self.loading_window.update_progress(95, "Loading complete...", "")

            nan_count = results.get("nan_count", 0)
            if nan_count > 0:
                self.status_label.setText(
                    f"Dataset analyzed: {len(self.dataset)} samples, "
                    f"{len(self.dataset.columns)} features, {nan_count} NaN values"
                )
            else:
                self.status_label.setText(
                    f"Dataset analyzed: {len(self.dataset)} samples, "
                    f"{len(self.dataset.columns)} features"
                )

            # Show the real window
            debug_log("Setup complete, showing real window")
            print("Setup complete, showing real window")
            sys.stdout.flush()
            self.loading_window.update_progress(100, "Ready", "")
            QCoreApplication.processEvents()
            self.loading_window.show_real_window(self)
            print("**** UI FULLY LOADED ****")
            sys.stdout.flush()

        except Exception as e:
            debug_log(f"ERROR setting up UI: {str(e)}")
            print(f"ERROR SETTING UP UI: {str(e)}")
            sys.stdout.flush()
            traceback.print_exc()
            self.loading_window.update_progress(100, "Error", str(e))

            # Show error message
            QMessageBox.critical(
                self.loading_window,
                "Setup Error",
                f"Error setting up ML Model window:\n{str(e)}",
            )

            # Close the loading window
            self.loading_window.close()

    def update_column_selectors(self):
        """Update input/target column selectors"""
        debug_log("Updating column selectors - START")
        try:
            # Clear existing items
            debug_log("Clearing existing combo items")
            self.input_features_combo.clear()
            self.target_features_combo.clear()
            QCoreApplication.processEvents()  # Keep UI responsive

            # Add grouped input options - just a few main categories to simplify
            debug_log("Adding grouped input options")
            if self.insole_cols:
                self.input_features_combo.addItem("Insole Forces (all)")
                debug_log(f"Added insole group with {len(self.insole_cols)} columns")

            if self.opencap_cols:
                self.input_features_combo.addItem("OpenCap Kinematics (all)")
                debug_log(f"Added OpenCap group with {len(self.opencap_cols)} columns")

            if self.qtm_force_cols:
                self.input_features_combo.addItem("QTM Forces (all)")
                debug_log(
                    f"Added QTM force group with {len(self.qtm_force_cols)} columns"
                )

            if self.qtm_kine_cols:
                self.input_features_combo.addItem("QTM Kinematics (all)")
                debug_log(
                    f"Added QTM kinematics group with {len(self.qtm_kine_cols)} columns"
                )

            # Add combined options
            self.input_features_combo.addItem("All OpenCap + Insole Features")
            self.input_features_combo.addItem("All Features")
            QCoreApplication.processEvents()  # Keep UI responsive

            # For performance, don't add individual columns - they're rarely needed
            # and can cause slowdown with large datasets

            # Add target variable options
            debug_log("Adding target options")
            for col in self.moment_cols:
                self.target_features_combo.addItem(col)
                debug_log(f"Added target column: {col}")

            if len(self.moment_cols) > 1:
                self.target_features_combo.addItem("All Joint Moments")
                debug_log("Added 'All Joint Moments' option")

            debug_log("Column selectors updated successfully")
        except Exception as e:
            debug_log(f"ERROR updating column selectors: {str(e)}")
            traceback.print_exc()

        QCoreApplication.processEvents()  # Final UI update

    # Model preparation and training methods
    def prepare_training_data(self):
        """Prepare the data for model training"""
        # Call the model manager to prepare the data
        result = self.model_manager.prepare_training_data()
        
        # The visualization will be updated by the TrainingTab component
        return result

    def train_model(self):
        """Train the CNN model"""
        return self.model_manager.train_model()

    def update_training_progress(self, epoch):
        """Update the training progress bar"""
        self.progress_bar.setValue(epoch)
        self.training_status.setText(
            f"Training progress: Epoch {epoch}/{self.epochs.value()}"
        )

        # Update status label as well
        self.status_label.setText(
            f"Training in progress: Epoch {epoch}/{self.epochs.value()}"
        )

    def update_training_history(self, history, current_epoch, max_epochs):
        """Update the training history plots in real-time"""
        try:
            from ..ml import ModelVisualizer

            ModelVisualizer.plot_training_history(
                self.history_canvas, history, current_epoch, max_epochs
            )
        except Exception as e:
            debug_log(f"Error updating training history: {str(e)}")
            traceback.print_exc()

    def training_finished(self, model):
        """Handle training completion"""
        debug_log("Training finished callback")
        self.model = model
        self.training_thread = None

        # Update UI
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.prepare_data_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.training_status.setText("Training complete!")

        # Enable evaluation and visualization buttons
        self.evaluate_btn.setEnabled(True)
        self.visualize_btn.setEnabled(True)
        self.visualize_architecture_btn.setEnabled(True)
        self.save_model_btn.setEnabled(True)

        # Plot training history
        self.visualization_manager.plot_training_history()

        # Automatically evaluate on test data
        self.evaluate_model()

    def training_error(self, error_msg):
        """Handle training error"""
        QMessageBox.critical(
            self, "Training Error", f"Error during training: {error_msg}"
        )
        self.reset_training_ui()

        # Update both status displays
        error_status = f"Training failed: {error_msg}"
        self.training_status.setText(error_status)
        self.status_label.setText(error_status)

    def stop_training(self):
        """Stop the training process"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.reset_training_ui()

            # Update both status displays
            stop_message = "Training stopped by user"
            self.training_status.setText(stop_message)
            self.status_label.setText(stop_message)

    def reset_training_ui(self):
        """Reset the training UI elements"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.prepare_data_btn.setEnabled(True)

    # Model evaluation and visualization methods
    def evaluate_model(self):
        """Evaluate the model on test data"""
        return self.visualization_manager.evaluate_model()

    def visualize_predictions(self):
        """Visualize model predictions against ground truth"""
        return self.visualization_manager.visualize_predictions()

    def visualize_model_architecture(self):
        """Visualize the model architecture using Netron"""
        return self.visualization_manager.visualize_model_architecture()

    # Model save/load methods
    def save_model(self):
        """Save the trained model to a file"""
        return self.model_manager.save_model()

    def load_model(self):
        """Load a trained model from a file"""
        return self.model_manager.load_model()

    def export_predictions(self):
        """Export predictions and performance metrics to CSV"""
        return self.visualization_manager.export_predictions()


def show_ml_model_window(dataset, parent=None):
    """
    Show the ML model window for training and visualizing models

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to use for training
    parent : QWidget, optional
        Parent widget

    Returns
    -------
    MLModelWindow
        The created window instance
    """
    try:
        window = MLModelWindow(dataset, parent)
        return window
    except Exception as e:
        traceback.print_exc()

        # Show error dialog
        from PyQt5.QtWidgets import QMessageBox

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(f"Error showing ML model window: {str(e)}")
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        return None
