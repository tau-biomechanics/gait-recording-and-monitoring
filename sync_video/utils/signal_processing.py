"""
Signal processing utility functions
"""

import numpy as np
from scipy import signal


def filter_signal(data, fs=100, cutoff=10, order=4):
    """
    Apply a low-pass Butterworth filter to smooth signals
    
    Parameters:
    -----------
    data : array-like
        Input signal to filter
    fs : float, optional
        Sampling frequency in Hz (default: 100)
    cutoff : float, optional
        Cutoff frequency in Hz (default: 10)
    order : int, optional
        Filter order (default: 4)
        
    Returns:
    --------
    array-like
        Filtered signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)

    # Handle NaN values
    valid_mask = ~np.isnan(data)
    filtered_data = np.copy(data)

    if np.sum(valid_mask) > 0:  # Only filter if there's valid data
        filtered_data[valid_mask] = signal.filtfilt(b, a, data[valid_mask])

    return filtered_data