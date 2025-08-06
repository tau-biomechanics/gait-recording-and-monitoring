"""
Data loaders for different file formats
"""

from .opencap_loader import load_opencap_data
from .qtm_loader import load_qtm_data
from .insole_loader import load_insole_data

__all__ = ['load_opencap_data', 'load_qtm_data', 'load_insole_data']