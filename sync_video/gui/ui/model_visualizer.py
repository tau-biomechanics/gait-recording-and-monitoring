"""
Real-time model architecture visualization for training tab using Netron
"""

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QLabel,
        QMessageBox,
        QProgressBar,
    )
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtCore import Qt, QUrl, QTemporaryDir
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import os
import tempfile
import json
import traceback
import time
import sys
import threading
import netron
from ..utils import debug_log


class NetronModelVisualizer(QWidget):
    """Widget for visualizing model architecture using Netron"""
    
    def __init__(self, parent=None):
        super(NetronModelVisualizer, self).__init__(parent)
        self.parent = parent
        self.netron_address = None
        self.temp_dir = None
        self.temp_model_path = None
        self.model = None
        
        # Set up the layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to maximize space
        
        # Create an invisible status bar for messages (will be shown only on errors)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setVisible(False)
        self.layout.addWidget(self.status_label)
        
        # Create an invisible progress bar (will be shown only during loading)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        
        # Create the web view for embedding Netron - give it maximum space
        try:
            self.web_view = QWebEngineView()
            self.layout.addWidget(self.web_view, 1)  # Add stretch factor of 1 to take all space
        except Exception as e:
            debug_log(f"Error creating web view: {str(e)}")
            self.status_label.setText(f"Error creating web view: {str(e)}")
            self.status_label.setVisible(True)
            self.web_view = None
        
        # Initialize
        self._initialize_temp_directory()
        
    def _initialize_temp_directory(self):
        """Initialize temporary directory for model files"""
        try:
            # Show progress during initialization
            self.progress_bar.setValue(10)
            self.progress_bar.setVisible(True)
            
            # Create a temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="netron_model_")
            self.temp_model_path = os.path.join(self.temp_dir, "model.h5")
            debug_log(f"Created temporary directory: {self.temp_dir}")
            
            # Start with a basic model
            self._generate_initial_model()
            
        except Exception as e:
            debug_log(f"Error initializing temp directory: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setVisible(True)
            self.progress_bar.setVisible(False)
    
    def _generate_initial_model(self):
        """Generate an initial TensorFlow model for visualization"""
        try:
            # Update progress
            self.progress_bar.setValue(25)
            
            # Import TensorFlow lazily to avoid loading it until needed
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Dense, Conv1D, MaxPooling1D, Flatten, 
                BatchNormalization, Dropout, InputLayer
            )
            
            # Create a basic model
            model = Sequential()
            model.add(InputLayer(input_shape=(20, 10)))
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(3, activation='linear'))
            
            # Compile the model (required to save it)
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Save the model
            model.save(self.temp_model_path)
            
            # Store the model for future updates
            self.model = model
            
            # Start Netron server
            self._start_netron_server()
            
        except Exception as e:
            debug_log(f"Error generating initial model: {str(e)}")
            traceback.print_exc()
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setVisible(True)
            self.progress_bar.setVisible(False)
    
    def _start_netron_server(self):
        """Start the Netron server"""
        try:
            # Show progress bar during loading
            self.progress_bar.setValue(50)
            self.progress_bar.setVisible(True)
            
            # Stop any existing server
            self.stop_server()
            
            # Start the server
            try:
                address = netron.start(self.temp_model_path, browse=False)
                debug_log(f"Netron returned address: {address} (type: {type(address)})")
                
                # Handle different address formats returned by netron.start()
                if isinstance(address, tuple):
                    # For newer versions, it might return (host, port)
                    host, port = address
                    self.netron_address = f"http://{host}:{port}"
                elif isinstance(address, int):
                    # It might just return a port
                    self.netron_address = f"http://localhost:{address}"
                else:
                    # It might return a full URL as string
                    self.netron_address = address
                    
                debug_log(f"Netron server started at {self.netron_address}")
                
                # Update the web view
                self._update_web_view()
                
            except Exception as e:
                debug_log(f"Error starting Netron server: {str(e)}")
                self.status_label.setText(f"Error: {str(e)}")
                self.status_label.setVisible(True)
                self.progress_bar.setVisible(False)
                
        except Exception as e:
            debug_log(f"Error starting Netron server: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setVisible(True)
            self.progress_bar.setVisible(False)
    
    def _update_web_view(self):
        """Update the web view to show the Netron visualization"""
        if not self.web_view or not self.netron_address:
            self.status_label.setText("Error: Web view or Netron server not available")
            self.status_label.setVisible(True)
            return
        
        try:
            # Show the progress bar while loading
            self.progress_bar.setValue(75)
            self.progress_bar.setVisible(True)
            
            # Ensure the URL is properly formatted
            url_str = self.netron_address
            
            debug_log(f"Loading URL: {url_str} (type: {type(url_str)})")
            
            # Make sure we have a string URL starting with http://
            if not isinstance(url_str, str):
                url_str = str(url_str)
            
            if not url_str.startswith('http'):
                if url_str.isdigit() or isinstance(url_str, int):
                    # It's just a port number
                    url_str = f"http://localhost:{url_str}"
                else:
                    # Add http:// prefix if missing
                    url_str = f"http://{url_str}"
            
            # Create proper QUrl and load it
            debug_log(f"Final URL to load: {url_str}")
            url = QUrl(url_str)
            
            # Check if URL is valid
            if not url.isValid():
                debug_log(f"Invalid URL: {url_str}")
                self.status_label.setText(f"Error: Invalid URL {url_str}")
                self.status_label.setVisible(True)
                return
                
            self.web_view.load(url)
            
            # Set progress to complete
            self.progress_bar.setValue(100)
            
            # Hide UI elements after a short delay
            # Using simple approach since we don't have QTimer in this context
            def hide_progress():
                time.sleep(0.5)  # Wait half a second
                self.progress_bar.setVisible(False)
                self.status_label.setVisible(False)
                
            # Run in a background thread to not block the UI
            threading.Thread(target=hide_progress, daemon=True).start()
            
        except Exception as e:
            debug_log(f"Error updating web view: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setVisible(True)
    
    def update_visualization(self, parent):
        """Update the visualization based on current model parameters"""
        try:
            # Show progress bar for loading
            self.progress_bar.setValue(25)
            self.progress_bar.setVisible(True)
            
            # Get current values from parent's UI
            seq_length = parent.seq_length.value() if hasattr(parent, 'seq_length') else 20
            n_features = 10  # Default, will be updated with actual data when available
            n_outputs = 3    # Default, will be updated with actual data when available
            
            # If we have actual data prepared, get the real feature/output counts
            if hasattr(parent, 'training_data') and parent.training_data:
                if 'input_cols' in parent.training_data:
                    n_features = len(parent.training_data['input_cols'])
                if 'target_cols' in parent.training_data:
                    n_outputs = len(parent.training_data['target_cols'])
            
            conv_layers = parent.conv_layers.value()
            filters = parent.filters.value()
            dense_layers = parent.dense_layers.value()
            use_sequences = parent.use_sequences.isChecked() if hasattr(parent, 'use_sequences') else True
            
            # Generate updated model
            self._generate_model(
                seq_length, n_features, n_outputs,
                conv_layers, filters, dense_layers,
                use_sequences
            )
            
            # Save the model
            self.model.save(self.temp_model_path)
            
            # Restart the Netron server
            self._start_netron_server()
            
        except Exception as e:
            debug_log(f"Error updating model visualization: {str(e)}")
            traceback.print_exc()
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setVisible(True)
            self.progress_bar.setVisible(False)
    
    def _generate_model(self, seq_length, n_features, n_outputs, 
                       conv_layers, filters, dense_layers, use_sequences=True):
        """Generate a new TensorFlow model based on the parameters"""
        try:
            # Import TensorFlow lazily to avoid loading it until needed
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Dense, Conv1D, MaxPooling1D, Flatten, 
                BatchNormalization, Dropout, InputLayer
            )
            
            # Create a new model
            model = Sequential()
            
            # Input layer
            if use_sequences:
                model.add(InputLayer(input_shape=(seq_length, n_features)))
                
                # Add convolutional and pooling layers
                for i in range(conv_layers):
                    # Conv layer
                    current_filters = filters if i == 0 else filters * 2
                    model.add(Conv1D(
                        filters=current_filters, 
                        kernel_size=3, 
                        activation='relu',
                        padding='same'
                    ))
                    model.add(BatchNormalization())
                    model.add(MaxPooling1D(pool_size=2))
                    model.add(Dropout(0.2))
                
                # Flatten layer
                model.add(Flatten())
            else:
                # For direct model, just add a single input layer
                model.add(InputLayer(input_shape=(n_features,)))
            
            # Dense layers
            for i in range(dense_layers):
                model.add(Dense(64, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
            
            # Output layer
            model.add(Dense(n_outputs, activation='linear'))
            
            # Compile the model (required to save it)
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Store the model
            self.model = model
            
            return model
            
        except Exception as e:
            debug_log(f"Error generating model: {str(e)}")
            traceback.print_exc()
            raise
    
    def stop_server(self):
        """Stop the Netron server"""
        try:
            if self.netron_address:
                debug_log(f"Attempting to stop Netron server at {self.netron_address}")
                
                # Handle different address formats
                address = self.netron_address
                
                # If it's a URL, try to extract the port
                if isinstance(address, str) and address.startswith('http'):
                    # Try to extract port number
                    try:
                        # http://localhost:8080 -> 8080
                        port = int(address.split(':')[-1].split('/')[0])
                        netron.stop(port)
                    except:
                        # Try stopping with the whole URL
                        netron.stop(address)
                else:
                    # Otherwise use the address directly
                    netron.stop(address)
                    
                self.netron_address = None
                debug_log("Netron server stopped")
        except Exception as e:
            debug_log(f"Error stopping Netron server: {str(e)}")
            
    def closeEvent(self, event):
        """Handle close event to stop Netron server"""
        self.stop_server()
        super(NetronModelVisualizer, self).closeEvent(event)