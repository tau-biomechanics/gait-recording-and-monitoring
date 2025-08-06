"""
Loading window for displaying progress during ML model initialization
"""

try:
    from PyQt5.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QLabel,
        QProgressBar,
    )
    from PyQt5.QtCore import QCoreApplication
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

from ..utils import debug_log


class SimpleLoadingWindow(QMainWindow):
    """
    A simple loading window that will be shown immediately while
    the full ML model window is being initialized in the background
    """

    def __init__(self, dataset, parent=None):
        debug_log("Creating SimpleLoadingWindow")
        super(SimpleLoadingWindow, self).__init__(parent)
        self.setWindowTitle("ML Model Loading")
        self.setMinimumSize(400, 200)
        self.dataset_size = len(dataset)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Loading message
        layout.addWidget(
            QLabel(f"Loading ML Model Window for {self.dataset_size} data points...")
        )
        layout.addWidget(QLabel("This may take a few moments."))

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Status label
        self.status = QLabel("Initializing...")
        layout.addWidget(self.status)

        # Detailed status
        self.detail_status = QLabel("")
        layout.addWidget(self.detail_status)

        # Real window reference
        self.real_window = None
        debug_log("SimpleLoadingWindow created")

    def update_progress(self, value, message, detail=""):
        """Update progress bar and status message"""
        debug_log(f"Loading progress: {value}% - {message} - {detail}")
        self.progress.setValue(value)
        self.status.setText(message)
        if detail:
            self.detail_status.setText(detail)
        # Force UI update
        QCoreApplication.processEvents()

    def show_real_window(self, window):
        """Show the real window and hide this one"""
        debug_log("Showing real window")
        self.real_window = window
        window.show()
        self.close()
        debug_log("Loading window closed")