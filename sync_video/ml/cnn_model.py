"""
1D CNN model for predicting joint moments from biomechanical data
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tempfile
from datetime import datetime


class JointMomentCNN:
    """
    1D Convolutional Neural Network for predicting joint moments from biomechanical data
    """
    
    def __init__(self, input_length=100, n_features=1, n_outputs=1):
        """
        Initialize the CNN model
        
        Parameters
        ----------
        input_length : int
            Length of input time series sequences
        n_features : int
            Number of input features
        n_outputs : int
            Number of output variables to predict
        """
        self.input_length = input_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.model = None
        self.history = None
        self.model_path = None
        self.scaler_x = None
        self.scaler_y = None
    
    def build_model(self, conv_layers=2, filters=64, kernel_size=3, dense_layers=2, dense_units=64, 
                   learning_rate=0.001, clipnorm=1.0):
        """
        Build the CNN model architecture
        
        Parameters
        ----------
        conv_layers : int
            Number of convolutional layers
        filters : int
            Number of filters in convolutional layers
        kernel_size : int
            Size of convolutional kernels
        dense_layers : int
            Number of dense layers
        dense_units : int
            Number of units in dense layers
        learning_rate : float
            Learning rate for the optimizer
        clipnorm : float
            Gradient clipping norm value to prevent exploding gradients
        """
        model = Sequential()
        
        # Input layer
        model.add(Conv1D(filters=filters, 
                         kernel_size=kernel_size, 
                         activation='relu', 
                         input_shape=(self.input_length, self.n_features)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        # Additional convolutional layers
        for i in range(conv_layers-1):
            model.add(Conv1D(filters=filters*2, 
                             kernel_size=kernel_size, 
                             activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
        
        # Flatten before dense layers
        model.add(Flatten())
        
        # Dense layers
        for i in range(dense_layers):
            model.add(Dense(dense_units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(self.n_outputs, activation='linear'))
        
        # Compile model with gradient clipping
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        model.compile(optimizer=optimizer, 
                     loss='mse', 
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_sequences(self, X, y, window_size=None):
        """
        Prepare time series data into sequences for training,
        with checks to remove NaN values
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        window_size : int, optional
            Size of sliding window, defaults to input_length
            
        Returns
        -------
        tuple
            (X_seq, y_seq) - Prepared sequences with NaN values removed
        """
        print(f"Preparing sequences with window size: {window_size}")
        
        if window_size is None:
            window_size = self.input_length
        
        # Adjust window size if it's too big
        if window_size > len(X) // 3:
            new_window_size = max(5, len(X) // 5)
            print(f"WARNING: Window size {window_size} is too large for dataset with {len(X)} samples")
            print(f"Adjusting window size to {new_window_size}")
            window_size = new_window_size
        
        # Check for and report NaN values
        nan_in_X = np.isnan(X).any()
        nan_in_y = np.isnan(y).any()
        
        if nan_in_X or nan_in_y:
            print(f"Warning: Found NaN values in input data (X: {nan_in_X}, y: {nan_in_y})")
            print(f"X shape: {X.shape}, NaN count: {np.isnan(X).sum()}")
            print(f"y shape: {y.shape}, NaN count: {np.isnan(y).sum()}")
            
            # Remove rows with NaN values in either X or y
            X_clean = X.copy()
            y_clean = y.copy()
            
            # Find indices of rows with NaN in X or y
            nan_mask_X = np.isnan(X).any(axis=1) if X.ndim > 1 else np.isnan(X)
            nan_mask_y = np.isnan(y).any(axis=1) if y.ndim > 1 else np.isnan(y)
            nan_mask = nan_mask_X | nan_mask_y
            
            # Remove rows with NaN values
            X_clean = X[~nan_mask]
            y_clean = y[~nan_mask]
            
            print(f"After NaN removal: X shape: {X_clean.shape}, y shape: {y_clean.shape}")
            
            X = X_clean
            y = y_clean
        
        # Check if we have enough data after filtering
        if len(X) <= window_size:
            print(f"ERROR: Not enough data points after NaN removal. Have {len(X)}, need more than {window_size}")
            # Use a smaller window size as a last resort
            last_resort_window = max(3, len(X) // 2 - 1)
            if last_resort_window < window_size:
                print(f"Trying with much smaller window size: {last_resort_window}")
                window_size = last_resort_window
            else:
                raise ValueError(f"Dataset too small ({len(X)} samples) for sequence creation even with reduced window size")
        
        print(f"Creating sequences with window size {window_size} from {len(X)} samples")
        
        # Create sequences with improved error handling
        X_seq = []
        y_seq = []
        failed_windows = 0
        
        for i in range(len(X) - window_size):
            # Check for NaN values in this window
            current_X = X[i:i + window_size]
            current_y = y[i + window_size]
            
            if not np.isnan(current_X).any() and not np.isnan(current_y).any():
                X_seq.append(current_X)
                y_seq.append(current_y)
            else:
                failed_windows += 1
        
        print(f"Created {len(X_seq)} sequences, discarded {failed_windows} windows with NaN values")
        
        if len(X_seq) == 0:
            print("ERROR: No valid sequences could be created after NaN filtering")
            print(f"Dataset info: X shape {X.shape}, y shape {y.shape}")
            print(f"NaN counts: X {np.isnan(X).sum()}, y {np.isnan(y).sum()}")
            
            # As a last resort, try to fill remaining NaNs with zeros
            X_filled = np.nan_to_num(X, nan=0.0)
            y_filled = np.nan_to_num(y, nan=0.0)
            
            X_seq = []
            y_seq = []
            
            # Create sequences with NaN-filled data
            for i in range(len(X_filled) - window_size):
                X_seq.append(X_filled[i:i + window_size])
                y_seq.append(y_filled[i + window_size])
            
            print(f"Last resort: Created {len(X_seq)} sequences with NaN values replaced with zeros")
            
            if len(X_seq) == 0:
                raise ValueError("No valid sequences could be created after NaN filtering")
        
        return np.array(X_seq), np.array(y_seq)
    
    def normalize_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Normalize the input and output data with robust handling of outliers
        
        Parameters
        ----------
        X_train : numpy.ndarray
            Training input features
        y_train : numpy.ndarray
            Training target values
        X_val : numpy.ndarray, optional
            Validation input features
        y_val : numpy.ndarray, optional
            Validation target values
            
        Returns
        -------
        tuple
            Normalized data (X_train_norm, y_train_norm, X_val_norm, y_val_norm)
        """
        from sklearn.preprocessing import RobustScaler
        
        # Initialize scalers if they don't exist
        if self.scaler_x is None:
            # Use RobustScaler which is less sensitive to outliers
            self.scaler_x = RobustScaler()
            
            # Reshape to 2D for scaling
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            
            # Remove any remaining NaN values for fitting
            X_train_flat_clean = X_train_flat[~np.isnan(X_train_flat).any(axis=1)]
            if len(X_train_flat_clean) == 0:
                raise ValueError("No valid data points for scaling inputs after NaN removal")
                
            self.scaler_x.fit(X_train_flat_clean)
        
        if self.scaler_y is None:
            # Use RobustScaler for targets too
            self.scaler_y = RobustScaler()
            
            # Reshape to 2D for scaling
            y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
            
            # Remove any NaN values for fitting
            y_train_clean = y_train_reshaped[~np.isnan(y_train_reshaped).any(axis=1)]
            if len(y_train_clean) == 0:
                raise ValueError("No valid data points for scaling targets after NaN removal")
                
            self.scaler_y.fit(y_train_clean)
        
        # Function to safely normalize data with NaN handling
        def safe_transform(scaler, data, is_3d=False):
            if is_3d:
                # Save original shape
                orig_shape = data.shape
                
                # Reshape to 2D
                data_flat = data.reshape(-1, data.shape[-1])
                
                # Create mask for NaN values
                nan_mask = np.isnan(data_flat).any(axis=1)
                
                # Transform non-NaN values
                data_flat_valid = data_flat[~nan_mask]
                data_flat_transformed = np.full_like(data_flat, np.nan)
                data_flat_transformed[~nan_mask] = scaler.transform(data_flat_valid)
                
                # Reshape back to original shape
                return data_flat_transformed.reshape(orig_shape)
            else:
                # Create mask for NaN values
                nan_mask = np.isnan(data).any(axis=1)
                
                # Transform non-NaN values
                data_transformed = np.full_like(data, np.nan)
                data_transformed[~nan_mask] = scaler.transform(data[~nan_mask])
                
                return data_transformed
        
        # Normalize training data
        X_train_norm = safe_transform(self.scaler_x, X_train, is_3d=True)
        y_train_norm = safe_transform(self.scaler_y, y_train.reshape(-1, y_train.shape[-1]))
        
        # Check for any remaining NaN values after normalization
        if np.isnan(X_train_norm).any() or np.isnan(y_train_norm).any():
            print("Warning: NaN values remain after normalization")
            print(f"X_train_norm NaN count: {np.isnan(X_train_norm).sum()}")
            print(f"y_train_norm NaN count: {np.isnan(y_train_norm).sum()}")
            
            # Remove sequences with NaN values
            valid_indices = ~(np.isnan(X_train_norm).any(axis=(1, 2)) | np.isnan(y_train_norm).any(axis=1))
            X_train_norm = X_train_norm[valid_indices]
            y_train_norm = y_train_norm[valid_indices]
            
            print(f"After final NaN removal: X_train_norm shape: {X_train_norm.shape}, y_train_norm shape: {y_train_norm.shape}")
        
        if X_val is not None and y_val is not None:
            # Normalize validation data
            X_val_norm = safe_transform(self.scaler_x, X_val, is_3d=True)
            y_val_norm = safe_transform(self.scaler_y, y_val.reshape(-1, y_val.shape[-1]))
            
            # Remove validation sequences with NaN values
            if np.isnan(X_val_norm).any() or np.isnan(y_val_norm).any():
                valid_indices = ~(np.isnan(X_val_norm).any(axis=(1, 2)) | np.isnan(y_val_norm).any(axis=1))
                X_val_norm = X_val_norm[valid_indices]
                y_val_norm = y_val_norm[valid_indices]
            
            return X_train_norm, y_train_norm, X_val_norm, y_val_norm
        
        return X_train_norm, y_train_norm
    
    def train(self, X_train, y_train, validation_data=None, epochs=100, batch_size=32, 
              patience=20, save_model=True):
        """
        Train the CNN model
        
        Parameters
        ----------
        X_train : numpy.ndarray
            Training input features
        y_train : numpy.ndarray
            Training target values
        validation_data : tuple, optional
            Tuple of (X_val, y_val) for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        patience : int
            Patience for early stopping
        save_model : bool
            Whether to save the best model
        
        Returns
        -------
        tensorflow.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Make one final check for NaN values before training
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("NaN values detected in training data. Please preprocess data first.")
        
        # Create a custom callback to monitor and log loss values
        class NaNLossCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs.get('loss') is None or np.isnan(logs.get('loss')):
                    print(f"\nWarning: NaN loss detected at epoch {epoch+1}. Stopping training.")
                    self.model.stop_training = True
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            NaNLossCallback()
        ]
        
        if save_model:
            # Create a timestamp for the model filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(tempfile.gettempdir(), 'joint_moment_models')
            os.makedirs(model_dir, exist_ok=True)
            
            self.model_path = os.path.join(model_dir, f'joint_moment_cnn_{timestamp}.h5')
            callbacks.append(
                ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
            )
        
        print(f"Training CNN model with {X_train.shape} input and {y_train.shape} output...")
        
        # Check validation data for NaN values
        if validation_data is not None:
            X_val, y_val = validation_data
            if np.isnan(X_val).any() or np.isnan(y_val).any():
                raise ValueError("NaN values detected in validation data. Please preprocess data first.")
        
        # Add a try-except block to catch and log numerical errors during training
        try:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            # Check if training worked (no NaN loss)
            if np.isnan(self.history.history['loss']).any():
                print("Warning: NaN values detected in training history. "
                      "Training may have been unstable.")
                
                # Find last non-NaN loss value
                valid_losses = [loss for loss in self.history.history['loss'] if not np.isnan(loss)]
                
                if valid_losses:
                    print(f"Last valid training loss: {valid_losses[-1]}")
                else:
                    print("No valid loss values found.")
            
            return self.history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, X, denormalize=True):
        """
        Make predictions with the trained model
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features
        denormalize : bool
            Whether to denormalize predictions
            
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Check for NaN values in input
        if np.isnan(X).any():
            print("Warning: NaN values detected in prediction input. Replacing with zeros.")
            X_clean = np.nan_to_num(X, nan=0.0)
            X = X_clean
        
        # Normalize input if scaler exists
        if self.scaler_x is not None:
            X_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_norm_flat = self.scaler_x.transform(X_flat)
            X_norm = X_norm_flat.reshape(X_shape)
        else:
            X_norm = X
        
        # Make predictions
        y_pred_norm = self.model.predict(X_norm)
        
        # Denormalize predictions if requested and scaler exists
        if denormalize and self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred_norm)
            return y_pred
        
        return y_pred_norm
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters
        ----------
        X_test : numpy.ndarray
            Test input features
        y_test : numpy.ndarray
            Test target values
            
        Returns
        -------
        tuple
            (loss, metrics) - Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Check for NaN values
        if np.isnan(X_test).any() or np.isnan(y_test).any():
            print("Warning: NaN values detected in test data. Cleaning data before evaluation.")
            
            # Create mask for rows without NaN values
            valid_indices = ~(np.isnan(X_test).any(axis=(1, 2)) | np.isnan(y_test).any(axis=1))
            
            if not np.any(valid_indices):
                raise ValueError("No valid data points for evaluation after NaN removal")
            
            X_test = X_test[valid_indices]
            y_test = y_test[valid_indices]
        
        # Normalize data if scalers exist
        if self.scaler_x is not None and self.scaler_y is not None:
            X_test_shape = X_test.shape
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            X_test_norm_flat = self.scaler_x.transform(X_test_flat)
            X_test_norm = X_test_norm_flat.reshape(X_test_shape)
            
            y_test_norm = self.scaler_y.transform(y_test.reshape(-1, y_test.shape[-1]))
            
            return self.model.evaluate(X_test_norm, y_test_norm)
        
        return self.model.evaluate(X_test, y_test)
    
    def plot_training_history(self):
        """
        Plot the training history
        
        Returns
        -------
        tuple
            (fig, axes) - Matplotlib figure and axes
        """
        if self.history is None:
            raise ValueError("Model not trained yet.")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Filter out any NaN values for plotting
        loss_history = np.array(self.history.history['loss'])
        val_loss_history = np.array(self.history.history['val_loss'])
        mae_history = np.array(self.history.history['mae'])
        val_mae_history = np.array(self.history.history['val_mae'])
        
        # Create epoch indices
        epochs = np.arange(1, len(loss_history) + 1)
        
        # Plot only non-NaN loss values
        valid_loss_mask = ~np.isnan(loss_history)
        valid_val_loss_mask = ~np.isnan(val_loss_history)
        
        if np.any(valid_loss_mask):
            axes[0].plot(epochs[valid_loss_mask], loss_history[valid_loss_mask], label='Training Loss')
        if np.any(valid_val_loss_mask):
            axes[0].plot(epochs[valid_val_loss_mask], val_loss_history[valid_val_loss_mask], label='Validation Loss')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot only non-NaN MAE values
        valid_mae_mask = ~np.isnan(mae_history)
        valid_val_mae_mask = ~np.isnan(val_mae_history)
        
        if np.any(valid_mae_mask):
            axes[1].plot(epochs[valid_mae_mask], mae_history[valid_mae_mask], label='Training MAE')
        if np.any(valid_val_mae_mask):
            axes[1].plot(epochs[valid_val_mae_mask], val_mae_history[valid_val_mae_mask], label='Validation MAE')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        fig.tight_layout()
        return fig, axes
    
    def visualize_predictions(self, X_test, y_test, time_values=None):
        """
        Visualize model predictions against ground truth
        
        Parameters
        ----------
        X_test : numpy.ndarray
            Test input features
        y_test : numpy.ndarray
            Test target values (ground truth)
        time_values : numpy.ndarray, optional
            Time values for x-axis
            
        Returns
        -------
        tuple
            (fig, axes) - Matplotlib figure and axes
        """
        # Check for and clean NaN values before prediction
        if np.isnan(X_test).any() or np.isnan(y_test).any():
            print("Warning: NaN values detected in visualization data. Cleaning data before visualization.")
            
            # Create mask for rows without NaN values
            valid_indices = ~(np.isnan(X_test).any(axis=(1, 2)) | np.isnan(y_test).any(axis=1))
            
            if not np.any(valid_indices):
                raise ValueError("No valid data points for visualization after NaN removal")
            
            X_test = X_test[valid_indices]
            y_test = y_test[valid_indices]
            
            if time_values is not None and len(time_values) > len(valid_indices):
                time_values = time_values[valid_indices]
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Create time values if not provided
        if time_values is None:
            time_values = np.arange(len(y_test))
        elif len(time_values) > len(y_test):
            time_values = time_values[:len(y_test)]
        
        # Determine how many outputs to plot
        n_outputs = y_test.shape[1]
        
        # Create a figure with subplots for each output
        fig, axes = plt.subplots(n_outputs, 1, figsize=(15, 4 * n_outputs))
        
        # If there's only one output, wrap axes in a list
        if n_outputs == 1:
            axes = [axes]
        
        # Plot each output
        for i in range(n_outputs):
            axes[i].plot(time_values, y_test[:, i], 'b-', label='Actual')
            axes[i].plot(time_values, y_pred[:, i], 'r-', label='Predicted')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(f'Output {i+1}')
            axes[i].set_title(f'Predicted vs Actual (Output {i+1})')
            axes[i].legend()
            axes[i].grid(True)
        
        fig.tight_layout()
        return fig, axes
    
    def save(self, path=None):
        """
        Save the model to a file
        
        Parameters
        ----------
        path : str, optional
            Path to save the model, if None, use the model_path
            
        Returns
        -------
        str
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if path is None:
            if self.model_path is None:
                # Create a timestamp for the model filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join(tempfile.gettempdir(), 'joint_moment_models')
                os.makedirs(model_dir, exist_ok=True)
                
                path = os.path.join(model_dir, f'joint_moment_cnn_{timestamp}.h5')
            else:
                path = self.model_path
        
        self.model.save(path)
        self.model_path = path
        
        # Save scalers if they exist
        if self.scaler_x is not None and self.scaler_y is not None:
            import joblib
            
            scaler_dir = os.path.dirname(path)
            scaler_x_path = os.path.join(scaler_dir, f"{os.path.basename(path).split('.')[0]}_scaler_x.joblib")
            scaler_y_path = os.path.join(scaler_dir, f"{os.path.basename(path).split('.')[0]}_scaler_y.joblib")
            
            joblib.dump(self.scaler_x, scaler_x_path)
            joblib.dump(self.scaler_y, scaler_y_path)
        
        return path
    
    @classmethod
    def load(cls, path):
        """
        Load a saved model
        
        Parameters
        ----------
        path : str
            Path to the saved model
            
        Returns
        -------
        JointMomentCNN
            Loaded model
        """
        import joblib
        
        # Load the Keras model
        keras_model = load_model(path)
        
        # Create a new instance
        instance = cls()
        instance.model = keras_model
        instance.model_path = path
        
        # Get model input shape
        instance.input_length = keras_model.input_shape[1]
        instance.n_features = keras_model.input_shape[2]
        instance.n_outputs = keras_model.output_shape[1]
        
        # Try to load scalers if they exist
        try:
            scaler_dir = os.path.dirname(path)
            scaler_x_path = os.path.join(scaler_dir, f"{os.path.basename(path).split('.')[0]}_scaler_x.joblib")
            scaler_y_path = os.path.join(scaler_dir, f"{os.path.basename(path).split('.')[0]}_scaler_y.joblib")
            
            if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
                instance.scaler_x = joblib.load(scaler_x_path)
                instance.scaler_y = joblib.load(scaler_y_path)
        except Exception as e:
            print(f"Could not load scalers: {e}")
        
        return instance


def prepare_data_for_training(dataset, input_cols, target_cols, test_size=0.2, validation_size=0.2, shuffle=True):
    """
    Prepare the dataset for training
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset containing input and target columns
    input_cols : list
        List of column names to use as inputs
    target_cols : list
        List of column names to use as targets
    test_size : float
        Proportion of data to use for testing
    validation_size : float
        Proportion of training data to use for validation
    shuffle : bool
        Whether to shuffle the data before splitting
        
    Returns
    -------
    dict
        Dictionary containing train, validation, and test data
    """
    from sklearn.model_selection import train_test_split
    
    # Check for NaN values in the dataset
    nan_cols = dataset[input_cols + target_cols].isna().sum()
    cols_with_nans = nan_cols[nan_cols > 0]
    if not cols_with_nans.empty:
        print("Warning: NaN values detected in dataset:")
        for col, count in cols_with_nans.items():
            print(f"  - {col}: {count} NaN values ({count/len(dataset)*100:.2f}%)")
    
    # Extract input and target data
    X = dataset[input_cols].values
    y = dataset[target_cols].values
    
    # Reshape y if it's a single target
    if len(target_cols) == 1:
        y = y.reshape(-1, 1)
    
    # Split into train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, shuffle=shuffle
    )
    
    # Get time values from the dataset if available
    time_col = dataset.columns[0] if 'time' in dataset.columns[0].lower() else None
    time_values = dataset[time_col].values if time_col else None
    
    # Create a dictionary with all the data
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'time_values': time_values,
        'input_cols': input_cols,
        'target_cols': target_cols
    }
    
    # Print summary of data shapes
    print(f"Data preparation complete:")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Validation set: {X_val.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    print(f"  - Input features: {len(input_cols)}")
    print(f"  - Target variables: {len(target_cols)}")
    
    return data 