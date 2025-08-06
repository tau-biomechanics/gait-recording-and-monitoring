"""
Thread for training CNN models without blocking the UI
"""

try:
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import traceback
from ..utils import debug_log


class TrainingThread(QThread):
    """Thread for training CNN models without blocking the UI"""

    # Define signals
    progress_signal = pyqtSignal(int)  # Training progress (epoch)
    history_signal = pyqtSignal(object, int, int)  # History data, current epoch, max epochs
    finished_signal = pyqtSignal(object)  # Training finished (model)
    error_signal = pyqtSignal(str)  # Error message

    def __init__(
        self, model, X_train, y_train, X_val, y_val, epochs, batch_size, patience
    ):
        debug_log("Creating TrainingThread")
        super(TrainingThread, self).__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        debug_log("TrainingThread created")

    def run(self):
        """Run the training process"""
        debug_log("Starting training thread")
        try:
            # Import tensorflow here to avoid loading it until needed
            debug_log("Importing TensorFlow")
            import tensorflow as tf

            debug_log(f"TensorFlow version: {tf.__version__}")

            # Custom callback to update progress and history in real-time
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_signal, history_signal, max_epochs):
                    self.progress_signal = progress_signal
                    self.history_signal = history_signal
                    self.max_epochs = max_epochs
                    self.current_history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

                def on_epoch_end(self, epoch, logs=None):
                    epoch_num = epoch + 1
                    
                    # Update progress bar
                    self.progress_signal.emit(epoch_num)
                    
                    # Update history with current epoch data
                    if logs:
                        # Add new values to our history collection
                        for key in logs:
                            if key in self.current_history:
                                self.current_history[key].append(logs[key])
                        
                        # Send the updated history for real-time plotting
                        self.history_signal.emit(self.current_history, epoch_num, self.max_epochs)

            # Create custom callback instance
            progress_cb = ProgressCallback(
                self.progress_signal, 
                self.history_signal,
                self.epochs
            )

            # Train the model with our custom callback
            debug_log("Starting model training")
            callbacks = [progress_cb]
            
            # Add early stopping
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=self.patience, 
                    restore_best_weights=True
                )
            )
            
            # Add model checkpoint if supported by the model
            if hasattr(self.model, 'model_path'):
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        self.model.model_path if self.model.model_path else 'model.h5',
                        monitor='val_loss',
                        save_best_only=True
                    )
                )
            
            # Train directly to get access to callback
            if hasattr(self.model, 'model'):
                # Use the Keras model directly for finer control
                self.model.history = self.model.model.fit(
                    self.X_train,
                    self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                # Use the model's train method
                self.model.train(
                    self.X_train,
                    self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    patience=self.patience,
                    save_model=True,
                )

            # Emit finished signal with the trained model
            debug_log("Training completed")
            self.finished_signal.emit(self.model)

        except Exception as e:
            debug_log(f"ERROR in training thread: {str(e)}")
            traceback.print_exc()
            self.error_signal.emit(str(e))