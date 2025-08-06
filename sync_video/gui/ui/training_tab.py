"""
Training tab component for the ML model window
"""

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QFormLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QProgressBar,
        QDoubleSpinBox,
        QSpinBox,
        QGroupBox,
        QSplitter,
        QFrame,
    )
    from PyQt5.QtCore import QCoreApplication, Qt
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import sys
import traceback
from ..utils import debug_log
from .model_visualizer import NetronModelVisualizer


class TrainingTab:
    """
    UI Component for ML model training tab
    """

    def __init__(self, parent):
        """
        Initialize the training tab

        Parameters
        ----------
        parent : MLModelWindow
            Parent window containing the tab
        """
        self.parent = parent
        self.layout = None
        self.model_visualizer = None

    def setup_ui(self):
        """Set up the training tab with model parameters and training controls"""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        self.layout = main_layout

        try:
            print("========== SETTING UP TRAINING TAB - START ==========")
            sys.stdout.flush()

            # Add a temporary status message
            status_label = QLabel("Setting up training controls...")
            main_layout.addWidget(status_label)
            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # Create left column for configuration and right column for visualization
            splitter = QSplitter(Qt.Horizontal)

            # ======== LEFT COLUMN - CONFIGURATION ========
            config_widget = QWidget()
            config_layout = QVBoxLayout(config_widget)

            # Data preparation group
            data_group = QGroupBox("Data Preparation")
            data_layout = QVBoxLayout(data_group)

            # Data selection
            print("Creating data selection controls")
            sys.stdout.flush()

            selection_layout = QGridLayout()
            selection_layout.addWidget(QLabel("Input Features:"), 0, 0)

            # Default feature options
            self.parent.input_features_combo = QComboBox()
            self.parent.input_features_combo.addItems(
                [
                    "All Input Features",
                    "Insole Features Only",
                    "OpenCap Features Only",
                    "QTM Force Features Only",
                    "QTM Kinematic Features Only",
                    "Custom Selection",
                ]
            )
            selection_layout.addWidget(self.parent.input_features_combo, 0, 1)

            selection_layout.addWidget(QLabel("Target Features:"), 1, 0)
            self.parent.target_features_combo = QComboBox()
            self.parent.target_features_combo.addItems(
                [
                    "All Moment Features",
                    "Ankle Moment Only",
                    "Knee Moment Only",
                    "Hip Moment Only",
                    "Custom Selection",
                ]
            )
            selection_layout.addWidget(self.parent.target_features_combo, 1, 1)

            data_layout.addLayout(selection_layout)
            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # Data splitting
            splitting_layout = QGridLayout()
            splitting_layout.addWidget(QLabel("Test Split:"), 0, 0)
            self.parent.test_split = QDoubleSpinBox()
            self.parent.test_split.setRange(0.1, 0.5)
            self.parent.test_split.setSingleStep(0.05)
            self.parent.test_split.setValue(0.2)
            splitting_layout.addWidget(self.parent.test_split, 0, 1)

            splitting_layout.addWidget(QLabel("Validation Split:"), 0, 2)
            self.parent.val_split = QDoubleSpinBox()
            self.parent.val_split.setRange(0.1, 0.5)
            self.parent.val_split.setSingleStep(0.05)
            self.parent.val_split.setValue(0.2)
            splitting_layout.addWidget(self.parent.val_split, 0, 3)

            splitting_layout.addWidget(QLabel("Shuffle Data:"), 1, 0)
            self.parent.shuffle_data = QCheckBox()
            self.parent.shuffle_data.setChecked(True)
            splitting_layout.addWidget(self.parent.shuffle_data, 1, 1)

            splitting_layout.addWidget(QLabel("Sequence Length:"), 1, 2)
            self.parent.seq_length = QSpinBox()
            self.parent.seq_length.setRange(5, 100)
            self.parent.seq_length.setValue(20)
            self.parent.seq_length.valueChanged.connect(self._update_visualization)
            splitting_layout.addWidget(self.parent.seq_length, 1, 3)

            # Add use sequences checkbox
            splitting_layout.addWidget(QLabel("Use Sequences:"), 2, 0)
            self.parent.use_sequences = QCheckBox()
            self.parent.use_sequences.setChecked(True)
            self.parent.use_sequences.setToolTip(
                "Enable to use sequence-based training (recommended for time series). Disable for small datasets."
            )
            self.parent.use_sequences.stateChanged.connect(self._update_visualization)
            splitting_layout.addWidget(self.parent.use_sequences, 2, 1)

            data_layout.addLayout(splitting_layout)

            # Add to configuration layout
            config_layout.addWidget(data_group)
            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # Model configuration group
            print("Creating model configuration controls")
            sys.stdout.flush()
            model_group = QGroupBox("Model Configuration")
            model_layout = QVBoxLayout(model_group)

            # Convolutional layers
            layers_layout = QGridLayout()
            layers_layout.addWidget(QLabel("Conv Layers:"), 0, 0)
            self.parent.conv_layers = QSpinBox()
            self.parent.conv_layers.setRange(1, 5)
            self.parent.conv_layers.setValue(2)
            self.parent.conv_layers.valueChanged.connect(self._update_visualization)
            layers_layout.addWidget(self.parent.conv_layers, 0, 1)

            layers_layout.addWidget(QLabel("Filters:"), 0, 2)
            self.parent.filters = QSpinBox()
            self.parent.filters.setRange(8, 128)
            self.parent.filters.setValue(64)
            self.parent.filters.valueChanged.connect(self._update_visualization)
            layers_layout.addWidget(self.parent.filters, 0, 3)

            layers_layout.addWidget(QLabel("Dense Layers:"), 1, 0)
            self.parent.dense_layers = QSpinBox()
            self.parent.dense_layers.setRange(1, 5)
            self.parent.dense_layers.setValue(2)
            self.parent.dense_layers.valueChanged.connect(self._update_visualization)
            layers_layout.addWidget(self.parent.dense_layers, 1, 1)

            model_layout.addLayout(layers_layout)
            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # Training parameters
            print("Creating training parameters")
            sys.stdout.flush()
            train_layout = QHBoxLayout()

            train_layout.addWidget(QLabel("Learning Rate:"))
            self.parent.learning_rate = QDoubleSpinBox()
            self.parent.learning_rate.setRange(0.0001, 0.1)
            self.parent.learning_rate.setDecimals(5)
            self.parent.learning_rate.setValue(0.001)
            self.parent.learning_rate.valueChanged.connect(self._update_visualization)
            train_layout.addWidget(self.parent.learning_rate)

            train_layout.addWidget(QLabel("Epochs:"))
            self.parent.epochs = QSpinBox()
            self.parent.epochs.setRange(10, 1000)
            self.parent.epochs.setValue(100)
            train_layout.addWidget(self.parent.epochs)

            train_layout.addWidget(QLabel("Batch Size:"))
            self.parent.batch_size = QSpinBox()
            self.parent.batch_size.setRange(8, 256)
            self.parent.batch_size.setValue(32)
            train_layout.addWidget(self.parent.batch_size)

            train_layout.addWidget(QLabel("Patience:"))
            self.parent.patience = QSpinBox()
            self.parent.patience.setRange(5, 50)
            self.parent.patience.setValue(20)
            train_layout.addWidget(self.parent.patience)

            model_layout.addLayout(train_layout)
            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # Add gradient clip norm
            print("Creating clipnorm control")
            sys.stdout.flush()
            self.parent.clipnorm = QDoubleSpinBox()
            self.parent.clipnorm.setRange(0.1, 10.0)
            self.parent.clipnorm.setDecimals(1)
            self.parent.clipnorm.setValue(1.0)
            self.parent.clipnorm.hide()  # Hide this control, just need the variable

            # Add to configuration layout
            print("Adding model group to main layout")
            sys.stdout.flush()
            config_layout.addWidget(model_group)
            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # Training progress - simpler layout
            print("Creating progress group")
            sys.stdout.flush()
            progress_group = QGroupBox("Training Progress")
            progress_layout = QVBoxLayout(progress_group)

            self.parent.progress_bar = QProgressBar()
            progress_layout.addWidget(self.parent.progress_bar)

            # Training status
            self.parent.training_status = QLabel("Ready to train")
            progress_layout.addWidget(self.parent.training_status)

            # Button layout
            print("Creating buttons")
            sys.stdout.flush()
            button_layout = QHBoxLayout()

            # Prepare data button
            self.parent.prepare_data_btn = QPushButton("Prepare Data")
            self.parent.prepare_data_btn.clicked.connect(self._on_prepare_data)
            button_layout.addWidget(self.parent.prepare_data_btn)

            # Train button
            self.parent.train_btn = QPushButton("Train Model")
            self.parent.train_btn.clicked.connect(self.parent.train_model)
            self.parent.train_btn.setEnabled(False)
            button_layout.addWidget(self.parent.train_btn)

            # Stop button
            self.parent.stop_btn = QPushButton("Stop Training")
            self.parent.stop_btn.clicked.connect(self.parent.stop_training)
            self.parent.stop_btn.setEnabled(False)
            button_layout.addWidget(self.parent.stop_btn)

            progress_layout.addLayout(button_layout)

            # Add to configuration layout
            print("Adding progress group to main layout")
            sys.stdout.flush()
            config_layout.addWidget(progress_group)

            # Add stretch to make sure UI elements are aligned to the top
            config_layout.addStretch(1)

            QCoreApplication.processEvents()  # Process events to keep UI responsive

            # ======== RIGHT COLUMN - VISUALIZATION ========
            # Create the Netron model visualizer
            print("Creating Netron model visualizer")
            sys.stdout.flush()
            self.model_visualizer = NetronModelVisualizer(self.parent)

            # Add both columns to the splitter
            splitter.addWidget(config_widget)
            splitter.addWidget(self.model_visualizer)

            # Set initial sizes (left: 40%, right: 60%)
            splitter.setSizes([400, 600])

            # Add splitter to main layout
            main_layout.removeWidget(status_label)
            status_label.setParent(None)
            main_layout.addWidget(splitter)

            # Update selectors
            print("Updating column selectors")
            sys.stdout.flush()
            self.parent.update_column_selectors()

            # Initial visualization update
            self._update_visualization()

            print("========== TRAINING TAB SETUP COMPLETE ==========")
            sys.stdout.flush()

        except Exception as e:
            print(f"ERROR in setting up training tab: {str(e)}")
            sys.stdout.flush()
            traceback.print_exc()
            # Keep a message about the error
            error_label = QLabel(f"Error setting up training tab: {str(e)}")
            error_label.setStyleSheet("color: red;")
            main_layout.addWidget(error_label)

        return tab

    def _update_visualization(self):
        """Update the model visualization when parameters change"""
        if self.model_visualizer:
            self.model_visualizer.update_visualization(self.parent)

    def _on_prepare_data(self):
        """Handle prepare data button click"""
        # Call the parent's prepare_training_data method
        result = self.parent.prepare_training_data()

        # Update visualization if data preparation was successful
        if result and self.model_visualizer:
            self._update_visualization()
