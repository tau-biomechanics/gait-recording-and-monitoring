"""
Visualization utilities for ML models and training results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import time


def debug_log(message):
    """Print a debug message with timestamp"""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DEBUG: {message}", flush=True)


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plot embedding in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        debug_log("Creating MatplotlibCanvas")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        debug_log("MatplotlibCanvas created")


class ModelVisualizer:
    """Class for visualizing model training and prediction results"""
    
    @staticmethod
    def create_history_canvas(parent=None, width=10, height=5, dpi=100):
        """
        Create a canvas for plotting training history
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        width : int
            Canvas width
        height : int
            Canvas height
        dpi : int
            Display resolution
            
        Returns
        -------
        MatplotlibCanvas
            Canvas for plotting training history
        """
        debug_log("Creating history canvas")
        history_canvas = MatplotlibCanvas(parent, width, height, dpi)
        history_canvas.fig.clear()
        history_canvas.fig.add_subplot(1, 2, 1)  # Left subplot for loss
        history_canvas.fig.add_subplot(1, 2, 2)  # Right subplot for MAE
        history_canvas.fig.tight_layout()
        return history_canvas
    
    @staticmethod
    def create_prediction_canvas(parent=None, n_outputs=1, width=10, height=None, dpi=100):
        """
        Create a canvas for plotting prediction results
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        n_outputs : int
            Number of outputs to visualize
        width : int
            Canvas width
        height : int, optional
            Canvas height (calculated from n_outputs if not provided)
        dpi : int
            Display resolution
            
        Returns
        -------
        MatplotlibCanvas
            Canvas for plotting predictions
        """
        debug_log(f"Creating prediction canvas for {n_outputs} outputs")
        # Calculate height based on number of outputs
        if height is None:
            height = 4 * n_outputs
        
        prediction_canvas = MatplotlibCanvas(parent, width, height, dpi)
        prediction_canvas.fig.clear()
        
        # Create subplots for each output
        for i in range(n_outputs):
            prediction_canvas.fig.add_subplot(n_outputs, 1, i + 1)
        
        prediction_canvas.fig.tight_layout()
        return prediction_canvas
    
    @staticmethod
    def plot_training_history(canvas, history, current_epoch=None, max_epochs=None):
        """
        Plot training history on the provided canvas
        
        Parameters
        ----------
        canvas : MatplotlibCanvas
            Canvas to plot on
        history : tf.keras.callbacks.History or dict
            Training history object or history dictionary
        current_epoch : int, optional
            Current epoch for real-time progress display
        max_epochs : int, optional
            Maximum number of epochs for real-time progress display
            
        Returns
        -------
        MatplotlibCanvas
            Updated canvas with training history plots
        """
        debug_log("Plotting training history")
        try:
            # Clear the canvas
            canvas.fig.clear()
            
            # Create subplots for loss and MAE
            ax1 = canvas.fig.add_subplot(1, 2, 1)
            ax2 = canvas.fig.add_subplot(1, 2, 2)
            
            # Handle both History objects and history dictionaries
            if hasattr(history, 'history'):
                history_dict = history.history
            else:
                history_dict = history
                
            # Filter out any NaN values for plotting
            loss_history = np.array(history_dict.get('loss', []))
            val_loss_history = np.array(history_dict.get('val_loss', []))
            mae_history = np.array(history_dict.get('mae', []))
            val_mae_history = np.array(history_dict.get('val_mae', []))
            
            # Create epoch indices
            epochs = np.arange(1, len(loss_history) + 1)
            
            # Plot real-time progress if specified
            if current_epoch is not None and max_epochs is not None:
                ax1.set_title(f'Training Loss (Epoch {current_epoch}/{max_epochs})')
                ax2.set_title(f'Training MAE (Epoch {current_epoch}/{max_epochs})')
            else:
                ax1.set_title('Training and Validation Loss')
                ax2.set_title('Training and Validation MAE')
            
            # Plot only non-NaN loss values
            valid_loss_mask = ~np.isnan(loss_history)
            valid_val_loss_mask = ~np.isnan(val_loss_history)
            
            if np.any(valid_loss_mask):
                ax1.plot(epochs[valid_loss_mask], loss_history[valid_loss_mask], 
                         'b-', label='Training Loss')
            if np.any(valid_val_loss_mask):
                ax1.plot(epochs[valid_val_loss_mask], val_loss_history[valid_val_loss_mask], 
                         'r-', label='Validation Loss')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (MSE)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot only non-NaN MAE values
            valid_mae_mask = ~np.isnan(mae_history)
            valid_val_mae_mask = ~np.isnan(val_mae_history)
            
            if np.any(valid_mae_mask):
                ax2.plot(epochs[valid_mae_mask], mae_history[valid_mae_mask], 
                         'b-', label='Training MAE')
            if np.any(valid_val_mae_mask):
                ax2.plot(epochs[valid_val_mae_mask], val_mae_history[valid_val_mae_mask], 
                         'r-', label='Validation MAE')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.legend()
            ax2.grid(True)
            
            # Apply tight layout and draw
            canvas.fig.tight_layout()
            canvas.draw()
            debug_log("Training history plot complete")
            
            return canvas
            
        except Exception as e:
            debug_log(f"Error plotting training history: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def plot_predictions(canvas, y_test, y_pred, time_values=None, target_names=None):
        """
        Plot model predictions against ground truth
        
        Parameters
        ----------
        canvas : MatplotlibCanvas
            Canvas to plot on
        y_test : numpy.ndarray
            Ground truth values
        y_pred : numpy.ndarray
            Predicted values
        time_values : numpy.ndarray, optional
            Time values for x-axis
        target_names : list, optional
            Names of the target variables
            
        Returns
        -------
        MatplotlibCanvas
            Updated canvas with prediction plots
        tuple
            Dictionary of metrics for each output
        """
        debug_log("Plotting predictions")
        try:
            # Clear the canvas
            canvas.fig.clear()
            
            # Create time values if not provided
            if time_values is None:
                time_values = np.arange(len(y_test))
            elif len(time_values) > len(y_test):
                time_values = time_values[:len(y_test)]
            
            # Determine number of outputs
            n_outputs = y_test.shape[1]
            
            # Calculate metrics for each output
            metrics = {}
            
            # Create subplots for each output
            for i in range(n_outputs):
                ax = canvas.fig.add_subplot(n_outputs, 1, i + 1)
                
                # Plot actual and predicted values
                ax.plot(time_values, y_test[:, i], 'b-', label='Actual')
                ax.plot(time_values, y_pred[:, i], 'r-', label='Predicted')
                
                # Calculate metrics for this output
                mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
                mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
                r2 = 1 - (
                    np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
                    / np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
                )
                
                metrics[i] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Get output name
                output_name = target_names[i] if target_names and i < len(target_names) else f"Output {i+1}"
                
                # Set axis labels and title
                ax.set_xlabel('Time')
                ax.set_ylabel(output_name)
                ax.set_title(f'Predicted vs Actual: {output_name}')
                
                # Add metrics as text annotation
                metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
                ax.text(
                    0.02, 0.95, metrics_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                
                # Add grid and legend
                ax.grid(True)
                ax.legend()
            
            # Apply tight layout and draw
            canvas.fig.tight_layout()
            canvas.draw()
            debug_log("Prediction plot complete")
            
            return canvas, metrics
            
        except Exception as e:
            debug_log(f"Error plotting predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, {}