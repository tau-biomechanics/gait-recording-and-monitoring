"""
Model manager for handling ML model operations
"""

import numpy as np
import pandas as pd
import traceback
from PyQt5.QtWidgets import QMessageBox

from .utils import debug_log


class ModelManager:
    """
    Manager for handling ML model operations
    """
    
    def __init__(self, parent):
        """
        Initialize the model manager
        
        Parameters
        ----------
        parent : MLModelWindow
            Parent window containing the ML model
        """
        self.parent = parent
        self.model = None
        
    def get_selected_inputs(self, input_selection):
        """
        Get selected input columns based on the input selection
        
        Parameters
        ----------
        input_selection : str
            Selection string from the input features combo box
            
        Returns
        -------
        list
            List of selected input column names
        """
        debug_log(f"Getting input features for: {input_selection}")
        
        try:
            # Handle predefined groups
            if input_selection == "All Input Features":
                return (
                    self.parent.insole_cols
                    + self.parent.opencap_cols
                    + self.parent.qtm_force_cols
                    + self.parent.qtm_kine_cols
                )
            elif input_selection == "Insole Features Only":
                return self.parent.insole_cols
            elif input_selection == "OpenCap Features Only":
                return self.parent.opencap_cols
            elif input_selection == "QTM Force Features Only":
                return self.parent.qtm_force_cols
            elif input_selection == "QTM Kinematic Features Only":
                return self.parent.qtm_kine_cols
            elif input_selection == "OpenCap + Insole":
                return self.parent.opencap_cols + self.parent.insole_cols
            elif input_selection in (
                self.parent.insole_cols
                + self.parent.opencap_cols
                + self.parent.qtm_force_cols
                + self.parent.qtm_kine_cols
            ):
                # Individual column selection
                return [input_selection]
            elif input_selection == "Custom Selection":
                # Not implemented yet - return all input features for now
                return (
                    self.parent.insole_cols
                    + self.parent.opencap_cols
                    + self.parent.qtm_force_cols
                    + self.parent.qtm_kine_cols
                )
            else:
                # Unknown selection, return all input features
                debug_log(f"Unknown input selection: {input_selection}, using all features")
                return (
                    self.parent.insole_cols
                    + self.parent.opencap_cols
                    + self.parent.qtm_force_cols
                    + self.parent.qtm_kine_cols
                )
        except Exception as e:
            debug_log(f"Error in get_selected_inputs: {str(e)}")
            traceback.print_exc()
            # Return all input features in case of error
            return (
                self.parent.insole_cols
                + self.parent.opencap_cols
                + self.parent.qtm_force_cols
                + self.parent.qtm_kine_cols
            )
            
    def get_selected_targets(self, target_selection):
        """
        Get selected target columns based on the target selection
        
        Parameters
        ----------
        target_selection : str
            Selection string from the target features combo box
            
        Returns
        -------
        list
            List of selected target column names
        """
        debug_log(f"Getting target features for: {target_selection}")
        
        try:
            # Handle predefined groups
            if target_selection == "All Moment Features":
                return self.parent.moment_cols
            elif target_selection == "Ankle Moment Only":
                # Find ankle moment columns
                ankle_moments = [col for col in self.parent.moment_cols if "ankle" in col.lower()]
                return ankle_moments if ankle_moments else self.parent.moment_cols
            elif target_selection == "Knee Moment Only":
                # Find knee moment columns
                knee_moments = [col for col in self.parent.moment_cols if "knee" in col.lower()]
                return knee_moments if knee_moments else self.parent.moment_cols
            elif target_selection == "Hip Moment Only":
                # Find hip moment columns
                hip_moments = [col for col in self.parent.moment_cols if "hip" in col.lower()]
                return hip_moments if hip_moments else self.parent.moment_cols
            elif target_selection in self.parent.moment_cols:
                # Individual column selection
                return [target_selection]
            elif target_selection == "Custom Selection":
                # Not implemented yet - return all moment features for now
                return self.parent.moment_cols
            else:
                # Unknown selection, return all moment features
                debug_log(f"Unknown target selection: {target_selection}, using all moments")
                return self.parent.moment_cols
        except Exception as e:
            debug_log(f"Error in get_selected_targets: {str(e)}")
            traceback.print_exc()
            # Return all moment features in case of error
            return self.parent.moment_cols

    def clean_dataset(self, dataset):
        """
        Clean the dataset by removing or replacing NaN values
        
        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataset to clean
            
        Returns
        -------
        pandas.DataFrame
            Cleaned dataset
        """
        try:
            debug_log("Cleaning dataset")
            
            # Make a copy of the dataset to avoid modifying the original
            clean_df = dataset.copy()
            
            # Check for NaN values
            nan_count = clean_df.isna().sum().sum()
            
            if nan_count > 0:
                debug_log(f"Found {nan_count} NaN values")
                
                # Always clean NaN values
                try:
                    # First try to interpolate
                    debug_log("Interpolating NaN values")
                    clean_df = clean_df.interpolate(
                        method="linear", limit_direction="both"
                    )
                    
                    # Check if any NaNs remain
                    remaining_nan_count = clean_df.isna().sum().sum()
                    debug_log(
                        f"Remaining NaN count after interpolation: {remaining_nan_count}"
                    )
                    
                    if remaining_nan_count > 0:
                        # Fill any remaining NaNs with zeros
                        debug_log("Filling remaining NaNs with zeros")
                        clean_df = clean_df.fillna(0.0)
                except Exception as e:
                    debug_log(f"Error during NaN cleaning: {str(e)}")
                    QMessageBox.warning(
                        self.parent,
                        "Warning",
                        f"Error during NaN cleaning: {str(e)}\nUsing original dataset."
                    )
                    return dataset  # Return original dataset if cleaning fails
            
            return clean_df
            
        except Exception as e:
            debug_log(f"ERROR cleaning dataset: {str(e)}")
            traceback.print_exc()
            return dataset  # Return original dataset if there's an error
            
    def prepare_training_data(self):
        """
        Prepare the data for model training
        
        Returns
        -------
        bool
            True if preparation was successful, False otherwise
        """
        try:
            # Import needed modules
            from ..ml import prepare_data_for_training, JointMomentCNN
            
            # Get selected columns
            input_selection = self.parent.input_features_combo.currentText()
            target_selection = self.parent.target_features_combo.currentText()
            
            print(f"Getting input columns for: '{input_selection}'")
            input_cols = self.get_selected_inputs(input_selection)
            print(f"Selected input columns: {input_cols}")
            
            print(f"Getting target columns for: '{target_selection}'")
            target_cols = self.get_selected_targets(target_selection)
            print(f"Selected target columns: {target_cols}")
            
            if not input_cols:
                QMessageBox.warning(self.parent, "Warning", "No input features selected")
                return False
                
            if not target_cols:
                QMessageBox.warning(self.parent, "Warning", "No target features selected")
                return False
            
            # Update status
            self.parent.training_status.setText("Cleaning dataset...")
            
            # Clean dataset if needed
            clean_dataset = self.clean_dataset(self.parent.dataset)
            if clean_dataset is None:
                QMessageBox.warning(self.parent, "Warning", "Dataset cleaning failed. Cannot proceed.")
                return False
            
            # Get data split parameters
            test_size = self.parent.test_split.value()
            val_size = self.parent.val_split.value()
            shuffle = self.parent.shuffle_data.isChecked()
            
            # Update status
            self.parent.training_status.setText("Preparing data...")
            
            # Make sure input_cols and target_cols are lists
            if isinstance(input_cols, str):
                input_cols = [input_cols]
            if isinstance(target_cols, str):
                target_cols = [target_cols]
                
            print(f"Input columns for training: {input_cols}")
            print(f"Target columns for training: {target_cols}")
            
            # Verify columns exist in the dataset
            missing_cols = [col for col in input_cols + target_cols if col not in clean_dataset.columns]
            if missing_cols:
                error_msg = f"Selected columns not found in dataset: {missing_cols}"
                print(f"ERROR: {error_msg}")
                QMessageBox.critical(self.parent, "Error", error_msg)
                return False
            
            # Prepare the data
            self.parent.training_data = prepare_data_for_training(
                clean_dataset, input_cols, target_cols, test_size, val_size, shuffle
            )
            
            # Create sequences
            window_size = self.parent.seq_length.value()
            n_features = len(input_cols)
            n_outputs = len(target_cols)
            
            # Update status
            self.parent.training_status.setText("Creating model...")
            
            # Check if sequences are enabled
            use_sequences = self.parent.use_sequences.isChecked()
            if use_sequences:
                # Initialize the model with gradient clipping for sequence data
                self.model = JointMomentCNN(
                    input_length=window_size, n_features=n_features, n_outputs=n_outputs
                )
            else:
                # Initialize the model for direct (non-sequence) training
                print("Direct training (no sequences) selected")
                # Use a very short sequence length of 1 for direct training
                self.model = JointMomentCNN(
                    input_length=1, n_features=n_features, n_outputs=n_outputs
                )
            
            # Store the model in the parent
            self.parent.model = self.model
            
            # Build the model with advanced parameters
            self.model.build_model(
                conv_layers=self.parent.conv_layers.value(),
                filters=self.parent.filters.value(),
                dense_layers=self.parent.dense_layers.value(),
                learning_rate=self.parent.learning_rate.value(),
                clipnorm=self.parent.clipnorm.value(),
            )
            
            # Update status
            self.parent.training_status.setText("Creating sequences...")
            
            # Create sequences for training if enabled
            if use_sequences:
                try:
                    X_train_seq, y_train_seq = self.model.prepare_sequences(
                        self.parent.training_data["X_train"],
                        self.parent.training_data["y_train"],
                        window_size,
                    )
                    
                    X_val_seq, y_val_seq = self.model.prepare_sequences(
                        self.parent.training_data["X_val"],
                        self.parent.training_data["y_val"],
                        window_size,
                    )
                    
                    X_test_seq, y_test_seq = self.model.prepare_sequences(
                        self.parent.training_data["X_test"],
                        self.parent.training_data["y_test"],
                        window_size,
                    )
                except ValueError as e:
                    QMessageBox.critical(
                        self.parent, "Error", f"Error creating sequences: {str(e)}"
                    )
                    return False
            else:
                # For direct (non-sequence) training, reshape the data
                # Each sample becomes a "sequence" of length 1
                print("Reshaping data for direct training (no sequences)")
                
                X_train_seq = self.parent.training_data["X_train"].reshape(-1, 1, n_features)
                y_train_seq = self.parent.training_data["y_train"]
                
                X_val_seq = self.parent.training_data["X_val"].reshape(-1, 1, n_features)
                y_val_seq = self.parent.training_data["y_val"]
                
                X_test_seq = self.parent.training_data["X_test"].reshape(-1, 1, n_features)
                y_test_seq = self.parent.training_data["y_test"]
                
                print(f"Reshaped training data: {X_train_seq.shape}, {y_train_seq.shape}")
            
            # Update status
            self.parent.training_status.setText("Normalizing data...")
            
            # Normalize the data
            try:
                X_train_norm, y_train_norm, X_val_norm, y_val_norm = (
                    self.model.normalize_data(
                        X_train_seq, y_train_seq, X_val_seq, y_val_seq
                    )
                )
            except ValueError as e:
                QMessageBox.critical(self.parent, "Error", f"Error normalizing data: {str(e)}")
                return False
            
            # Store the processed data
            self.parent.training_data["X_train_seq"] = X_train_seq
            self.parent.training_data["y_train_seq"] = y_train_seq
            self.parent.training_data["X_val_seq"] = X_val_seq
            self.parent.training_data["y_val_seq"] = y_val_seq
            self.parent.training_data["X_test_seq"] = X_test_seq
            self.parent.training_data["y_test_seq"] = y_test_seq
            
            self.parent.training_data["X_train_norm"] = X_train_norm
            self.parent.training_data["y_train_norm"] = y_train_norm
            self.parent.training_data["X_val_norm"] = X_val_norm
            self.parent.training_data["y_val_norm"] = y_val_norm
            
            # Update UI
            self.parent.train_btn.setEnabled(True)
            self.parent.training_status.setText(
                f"Data prepared: {len(X_train_norm)} training, {len(X_val_norm)} validation, "
                f"{len(X_test_seq)} test samples"
            )
            
            return True
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Error preparing data: {str(e)}")
            self.parent.training_status.setText(f"Error: {str(e)}")
            return False
            
    def train_model(self):
        """
        Train the CNN model
        
        Returns
        -------
        bool
            True if training started successfully, False otherwise
        """
        if self.model is None or self.parent.training_data is None:
            QMessageBox.warning(self.parent, "Warning", "Please prepare the data first")
            return False

        try:
            # Update UI
            self.parent.train_btn.setEnabled(False)
            self.parent.stop_btn.setEnabled(True)
            self.parent.prepare_data_btn.setEnabled(False)
            self.parent.progress_bar.setValue(0)
            self.parent.progress_bar.setMaximum(self.parent.epochs.value())
            self.parent.training_status.setText("Training in progress...")

            # Get training parameters
            epochs = self.parent.epochs.value()
            batch_size = self.parent.batch_size.value()
            patience = self.parent.patience.value()

            # Create training thread
            from .threads import TrainingThread
            self.parent.training_thread = TrainingThread(
                self.model,
                self.parent.training_data["X_train_norm"],
                self.parent.training_data["y_train_norm"],
                self.parent.training_data["X_val_norm"],
                self.parent.training_data["y_val_norm"],
                epochs,
                batch_size,
                patience,
            )

            # Connect signals
            self.parent.training_thread.progress_signal.connect(self.parent.update_training_progress)
            self.parent.training_thread.history_signal.connect(self.parent.update_training_history)
            self.parent.training_thread.finished_signal.connect(self.parent.training_finished)
            self.parent.training_thread.error_signal.connect(self.parent.training_error)

            # Switch to evaluation tab to see real-time progress
            self.parent.tabs.setCurrentIndex(1)

            # Start training
            self.parent.training_thread.start()
            return True

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Error starting training: {str(e)}")
            self.parent.reset_training_ui()
            return False
            
    def save_model(self):
        """
        Save the trained model to a file
        
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        if self.model is None:
            QMessageBox.warning(self.parent, "Warning", "No trained model available")
            return False

        try:
            # Get save path from user
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent, "Save Model", "", "Model Files (*.h5)"
            )

            if file_path:
                # Save the model
                save_path = self.model.save(file_path)
                QMessageBox.information(self.parent, "Success", f"Model saved to {save_path}")
                return True
            return False

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Error saving model: {str(e)}")
            return False
            
    def load_model(self):
        """
        Load a trained model from a file
        
        Returns
        -------
        bool
            True if load was successful, False otherwise
        """
        try:
            # Import the JointMomentCNN class
            from ..ml import JointMomentCNN
            from PyQt5.QtWidgets import QFileDialog

            # Get file path from user
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "Load Model", "", "Model Files (*.h5)"
            )

            if file_path:
                # Load the model
                self.model = JointMomentCNN.load(file_path)
                self.parent.model = self.model

                # Enable evaluation and visualization buttons
                self.parent.evaluate_btn.setEnabled(True)
                self.parent.visualize_btn.setEnabled(True)
                self.parent.visualize_architecture_btn.setEnabled(True)
                self.parent.save_model_btn.setEnabled(True)

                QMessageBox.information(
                    self.parent, "Success", f"Model loaded from {file_path}"
                )
                return True
            return False

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Error loading model: {str(e)}")
            return False