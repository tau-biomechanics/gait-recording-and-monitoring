"""
Evaluation tab component for the ML model window
"""

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QLabel,
        QPushButton,
    )
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import sys
import traceback


class EvaluationTab:
    """
    UI Component for ML model evaluation tab
    """
    
    def __init__(self, parent):
        """
        Initialize the evaluation tab
        
        Parameters
        ----------
        parent : MLModelWindow
            Parent window containing the tab
        """
        self.parent = parent
        
    def setup_ui(self):
        """Set up the evaluation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        try:
            print("========== SETTING UP EVALUATION TAB - START ==========")
            sys.stdout.flush()

            # Add a message
            layout.addWidget(
                QLabel("Evaluation features will be available after model training")
            )

            # Add placeholders for metrics
            self.parent.eval_metrics = QLabel("No model trained yet")
            layout.addWidget(self.parent.eval_metrics)
            
            # Also create metrics_label for backward compatibility (pointed to same label)
            self.parent.metrics_label = self.parent.eval_metrics
            
            # Create the history canvas for training history visualization in the evaluation tab
            from ...ml import ModelVisualizer
            self.parent.history_canvas = ModelVisualizer.create_history_canvas(self.parent)
            layout.addWidget(self.parent.history_canvas)

            # Add evaluation button (disabled for now)
            self.parent.evaluate_btn = QPushButton("Evaluate Model")
            self.parent.evaluate_btn.clicked.connect(self.parent.evaluate_model)
            self.parent.evaluate_btn.setEnabled(False)
            layout.addWidget(self.parent.evaluate_btn)

            print("========== EVALUATION TAB SETUP COMPLETE ==========")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"ERROR in setting up evaluation tab: {str(e)}")
            sys.stdout.flush()
            traceback.print_exc()
            # Keep a message about the error
            error_label = QLabel(f"Error setting up evaluation tab: {str(e)}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)
            
        return tab