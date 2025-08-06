"""
Visualization manager for handling ML model visualization
"""

import numpy as np
import pandas as pd
import traceback
import time
import os
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QFileDialog
from PyQt5.QtCore import QUrl

from .utils import debug_log


class VisualizationManager:
    """
    Manager for handling ML model visualization
    """
    
    def __init__(self, parent):
        """
        Initialize the visualization manager
        
        Parameters
        ----------
        parent : MLModelWindow
            Parent window containing the ML model
        """
        self.parent = parent
        
    def plot_training_history(self):
        """
        Plot the training history
        
        Returns
        -------
        bool
            True if plotting was successful, False otherwise
        """
        if self.parent.model and self.parent.model.history:
            try:
                # Use the ModelVisualizer to plot training history
                from ..ml import ModelVisualizer
                ModelVisualizer.plot_training_history(self.parent.history_canvas, self.parent.model.history)
                return True
            except Exception as e:
                print(f"Error plotting training history: {str(e)}")
                traceback.print_exc()
                return False
        return False
        
    def evaluate_model(self):
        """
        Evaluate the model on test data
        
        Returns
        -------
        bool
            True if evaluation was successful, False otherwise
        """
        if self.parent.model is None or self.parent.training_data is None:
            QMessageBox.warning(self.parent, "Warning", "No trained model available")
            return False

        try:
            # Evaluate on test data
            loss, mae = self.parent.model.evaluate(
                self.parent.training_data["X_test_seq"], self.parent.training_data["y_test_seq"]
            )

            # Update metrics display
            target_names = self.parent.training_data["target_cols"]
            metrics_text = f"Test Loss (MSE): {loss:.4f}\nTest MAE: {mae:.4f}\n\n"
            metrics_text += f"Target Variables: {', '.join(target_names)}"
            self.parent.eval_metrics.setText(metrics_text)

            # Visualize predictions on test data
            self.visualize_predictions()
            return True

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Error evaluating model: {str(e)}")
            return False
            
    def visualize_predictions(self):
        """
        Visualize model predictions against ground truth
        
        Returns
        -------
        bool
            True if visualization was successful, False otherwise
        """
        if self.parent.model is None or self.parent.training_data is None:
            QMessageBox.warning(self.parent, "Warning", "No trained model available")
            return False
        
        try:
            # Get test data
            X_test = self.parent.training_data["X_test_seq"]
            y_test = self.parent.training_data["y_test_seq"]
            
            # Get time values if available
            time_values = None
            if (
                "time_values" in self.parent.training_data
                and self.parent.training_data["time_values"] is not None
            ):
                # Use part of the time values corresponding to test data
                # This is approximate since we've transformed the data into sequences
                if len(self.parent.training_data["time_values"]) > len(y_test):
                    time_values = self.parent.training_data["time_values"][: len(y_test)]
            
            # Get predictions
            y_pred = self.parent.model.predict(X_test)
            
            # Get target names
            target_names = self.parent.training_data["target_cols"]
            
            # Use the ModelVisualizer to plot predictions
            from ..ml import ModelVisualizer
            _, metrics = ModelVisualizer.plot_predictions(
                self.parent.prediction_canvas, 
                y_test, 
                y_pred, 
                time_values, 
                target_names
            )
            
            # Create a table of metrics for all outputs
            metrics_table = "<table border='1' cellspacing='0' cellpadding='5' style='border-collapse: collapse;'>"
            metrics_table += "<tr><th>Output</th><th>MSE</th><th>MAE</th><th>R²</th></tr>"
            
            # Add metrics for each output
            for i, metric in metrics.items():
                name = target_names[i] if i < len(target_names) else f"Output {i+1}"
                metrics_table += f"<tr><td>{name}</td><td>{metric['mse']:.4f}</td><td>{metric['mae']:.4f}</td><td>{metric['r2']:.4f}</td></tr>"
            
            # Calculate overall metrics
            overall_mse = np.mean((y_test - y_pred) ** 2)
            overall_mae = np.mean(np.abs(y_test - y_pred))
            
            # Add overall metrics to the table
            metrics_table += f"<tr><td>Overall</td><td>{overall_mse:.4f}</td><td>{overall_mae:.4f}</td><td>-</td></tr>"
            metrics_table += "</table>"
            
            # Update metrics display
            self.parent.metrics_label.setText(metrics_table)
            
            # Enable buttons
            self.parent.visualize_btn.setEnabled(True)
            self.parent.export_btn.setEnabled(True)
            self.parent.save_model_btn.setEnabled(True)
            
            return True
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self.parent, "Error", f"Error visualizing predictions: {str(e)}"
            )
            return False
            
    def export_predictions(self):
        """
        Export predictions and performance metrics to CSV
        
        Returns
        -------
        bool
            True if export was successful, False otherwise
        """
        if self.parent.model is None or self.parent.training_data is None:
            QMessageBox.warning(self.parent, "Warning", "No trained model available")
            return False

        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent, "Export Predictions", "", "CSV Files (*.csv)"
            )

            if not file_path:
                return False

            # Get test data
            X_test = self.parent.training_data["X_test_seq"]
            y_test = self.parent.training_data["y_test_seq"]

            # Get predictions
            y_pred = self.parent.model.predict(X_test)

            # Create a DataFrame with actual and predicted values
            target_names = self.parent.training_data["target_cols"]

            # Initialize data dictionary
            data = {}

            # Get time values if available
            if (
                "time_values" in self.parent.training_data
                and self.parent.training_data["time_values"] is not None
            ):
                # Use part of the time values corresponding to test data
                if len(self.parent.training_data["time_values"]) > len(y_test):
                    time_values = self.parent.training_data["time_values"][: len(y_test)]
                    data["Time"] = time_values

            # Add actual and predicted values for each output
            for i, name in enumerate(target_names):
                data[f"{name}_Actual"] = y_test[:, i]
                data[f"{name}_Predicted"] = y_pred[:, i]

                # Calculate and add error
                data[f"{name}_Error"] = y_test[:, i] - y_pred[:, i]

                # Calculate and add metrics
                mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
                r2 = 1 - (
                    np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
                    / np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
                )

                # Add a note with metrics
                if i == 0:
                    data["Note"] = [
                        f"Metrics for {name}",
                        f"MSE: {mse:.6f}",
                        f"MAE: {mae:.6f}",
                        f"R²: {r2:.6f}",
                    ]
                    # Pad the rest of the rows
                    data["Note"].extend([""] * (len(y_test) - 4))

            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            QMessageBox.information(
                self.parent, "Export Successful", f"Predictions exported to {file_path}"
            )
            return True

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self.parent, "Error", f"Error exporting predictions: {str(e)}"
            )
            return False
            
    def visualize_model_architecture(self):
        """
        Visualize the model architecture using Netron
        
        Returns
        -------
        bool
            True if visualization was successful, False otherwise
        """
        if self.parent.model is None:
            QMessageBox.warning(self.parent, "Warning", "No trained model available")
            return False
            
        try:
            # Check if QWebEngineView is available
            try:
                from PyQt5.QtWebEngineWidgets import QWebEngineView
                use_webview = True
            except ImportError:
                use_webview = False
            
            # Create a temporary file to save the model
            import tempfile
            import netron
            
            # Create a temporary directory if needed
            temp_dir = os.path.join(tempfile.gettempdir(), 'joint_moment_models')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create a temp file for the model
            temp_model_path = os.path.join(temp_dir, f'temp_model_for_netron.h5')
            
            # Save the model to the temp file
            self.parent.model.save(temp_model_path)
            
            if use_webview:
                # Create a window for the visualization
                netron_window = QMainWindow(self.parent)
                netron_window.setWindowTitle("Model Architecture Visualization")
                netron_window.resize(1200, 800)
                
                # Create a web view to embed the Netron visualization
                web_view = QWebEngineView()
                netron_window.setCentralWidget(web_view)
                
                # Start the Netron server with the model
                netron_address = netron.start(temp_model_path, browse=False)
                
                # Load the Netron visualization in the web view
                web_view.load(QUrl(netron_address))
                
                # Show the window
                netron_window.show()
            else:
                # Fall back to opening in browser when QWebEngineView isn't available
                netron.start(temp_model_path)
                QMessageBox.information(
                    self.parent,
                    "Model Visualization",
                    "The model visualization has been opened in your web browser."
                )
            
            return True
                
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Error visualizing model architecture: {str(e)}")
            traceback.print_exc()
            return False