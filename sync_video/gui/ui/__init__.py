"""
UI components for the sync video application
"""

from .canvas import MatplotlibCanvas
from .loading_window import SimpleLoadingWindow
from .training_tab import TrainingTab
from .evaluation_tab import EvaluationTab
from .visualization_tab import VisualizationTab
from .model_visualizer import NetronModelVisualizer

__all__ = [
    "MatplotlibCanvas", 
    "SimpleLoadingWindow",
    "TrainingTab",
    "EvaluationTab",
    "VisualizationTab",
    "NetronModelVisualizer"
]