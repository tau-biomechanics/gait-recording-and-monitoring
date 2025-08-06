"""
Utility functions for the GUI components
"""

import time


def debug_log(message):
    """Print a debug message with timestamp"""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DEBUG: {message}", flush=True)