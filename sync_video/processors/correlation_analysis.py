"""
Correlation analysis for biomechanical time series data
"""

import numpy as np
from scipy import stats, signal
import pandas as pd


def calculate_similarity_metrics(series1, time1, series2, time2, resample=True):
    """
    Calculate similarity metrics between two time series.
    
    Parameters:
    -----------
    series1 : numpy.ndarray
        First time series data
    time1 : numpy.ndarray
        Time values for first series
    series2 : numpy.ndarray
        Second time series data
    time2 : numpy.ndarray
        Time values for second series
    resample : bool, optional
        Whether to resample series to match time points (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing calculated metrics
    """
    # Ensure data is in numpy array format
    series1 = np.array(series1)
    time1 = np.array(time1)
    series2 = np.array(series2)
    time2 = np.array(time2)
    
    # Print debug information
    print(f"Series 1 shape: {series1.shape}, Series 2 shape: {series2.shape}")
    print(f"Time 1 shape: {time1.shape}, Time 2 shape: {time2.shape}")
    
    # Handle NaN values by removing them
    valid_indices1 = ~np.isnan(series1)
    valid_indices2 = ~np.isnan(series2)
    
    series1 = series1[valid_indices1]
    time1 = time1[valid_indices1]
    series2 = series2[valid_indices2]
    time2 = time2[valid_indices2]
    
    # Check if we have enough data
    if len(series1) < 2 or len(series2) < 2:
        return {
            'error': 'Not enough valid data points for correlation analysis'
        }
    
    # Resample if time points don't match
    if resample and (len(time1) != len(time2) or not np.allclose(time1, time2)):
        # Create a common time vector (using the overlapping range with higher density)
        start_time = max(time1.min(), time2.min())
        end_time = min(time1.max(), time2.max())
        
        # Determine sampling rate (use the higher density one)
        density1 = len(time1) / (time1.max() - time1.min())
        density2 = len(time2) / (time2.max() - time2.min())
        
        # Use the higher density for resampling
        freq = max(density1, density2)
        num_points = int((end_time - start_time) * freq)
        
        # Create new time vector
        t_common = np.linspace(start_time, end_time, num_points)
        
        # Resample both series to the common time vector
        series1_resampled = np.interp(t_common, time1, series1)
        series2_resampled = np.interp(t_common, time2, series2)
        
        series1 = series1_resampled
        series2 = series2_resampled
        time_common = t_common
    else:
        time_common = time1
    
    # Scale the data if needed to make comparison meaningful
    # (e.g., if one series is in degrees and another in radians)
    if np.mean(series1) != 0 and np.mean(series2) != 0:
        scale_factor = np.abs(np.mean(series1) / np.mean(series2))
        
        # Only scale if the means are significantly different
        if scale_factor > 10 or scale_factor < 0.1:
            series2 = series2 * scale_factor
    
    # Calculate Pearson correlation coefficient
    pearson_r, pearson_p = stats.pearsonr(series1, series2)
    
    # Calculate Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(series1, series2)
    
    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((series1 - series2) ** 2))
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(series1 - series2))
    
    # Calculate Coefficient of Determination (R²)
    ss_res = np.sum((series2 - series1) ** 2)
    ss_tot = np.sum((series1 - np.mean(series1)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate cross-correlation with time lags
    cross_corr = signal.correlate(series1 - np.mean(series1), 
                                 series2 - np.mean(series2), 
                                 mode='full')
    
    # Normalize cross-correlation
    cross_corr = cross_corr / (np.std(series1) * np.std(series2) * len(series1))
    
    # Calculate time lags
    dt = np.mean(np.diff(time_common))
    lags = np.arange(-len(series1)+1, len(series1)) * dt
    
    # Find the lag with maximum correlation
    max_corr_idx = np.argmax(np.abs(cross_corr))
    max_lag = lags[max_corr_idx]
    max_corr = cross_corr[max_corr_idx]
    
    # Create time lag adjustment recommendation
    lag_recommendation = f"{max_lag:.3f}s" if np.abs(max_lag) > 0.01 else "No adjustment needed"
    lag_direction = "earlier" if max_lag > 0 else "later"
    if np.abs(max_lag) > 0.01:
        lag_recommendation = f"For optimal alignment, shift second series {np.abs(max_lag):.3f}s {lag_direction}"
    
    # Create summary report
    metrics = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'max_cross_correlation': max_corr,
        'optimal_lag': max_lag,
        'time_values': time_common,
        'series1_resampled': series1,
        'series2_resampled': series2,
        'cross_correlation': cross_corr,
        'lags': lags,
        'lag_recommendation': lag_recommendation
    }
    
    return metrics


def create_correlation_summary(metrics, series1_name, series2_name):
    """
    Create a formatted summary of correlation metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_similarity_metrics
    series1_name : str
        Name of the first data series
    series2_name : str
        Name of the second data series
        
    Returns:
    --------
    str
        Formatted summary text
    """
    if 'error' in metrics:
        return f"Error: {metrics['error']}"
    
    summary = f"Correlation Analysis: {series1_name} vs {series2_name}\n"
    summary += "=" * 50 + "\n\n"
    
    # Format Pearson correlation with significance stars
    p_stars = ""
    if metrics['pearson_p'] < 0.001:
        p_stars = "***"
    elif metrics['pearson_p'] < 0.01:
        p_stars = "**"
    elif metrics['pearson_p'] < 0.05:
        p_stars = "*"
    
    summary += f"Pearson Correlation (r): {metrics['pearson_r']:.4f} {p_stars}\n"
    summary += f"p-value: {metrics['pearson_p']:.6f}\n\n"
    
    summary += f"Spearman Rank Correlation: {metrics['spearman_r']:.4f}\n"
    summary += f"Coefficient of Determination (R²): {metrics['r_squared']:.4f}\n\n"
    
    summary += f"Root Mean Square Error (RMSE): {metrics['rmse']:.4f}\n"
    summary += f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n\n"
    
    summary += f"Maximum Cross-Correlation: {metrics['max_cross_correlation']:.4f}\n"
    summary += f"Optimal Time Lag: {metrics['optimal_lag']:.4f}s\n"
    summary += f"Recommendation: {metrics['lag_recommendation']}\n\n"
    
    # Interpret the correlation strength
    r_abs = abs(metrics['pearson_r'])
    if r_abs > 0.8:
        strength = "Very Strong"
    elif r_abs > 0.6:
        strength = "Strong"
    elif r_abs > 0.4:
        strength = "Moderate"
    elif r_abs > 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    summary += f"Correlation Strength: {strength}\n"
    
    if metrics['pearson_r'] > 0:
        direction = "Positive"
        interpretation = f"As {series1_name} increases, {series2_name} tends to increase."
    else:
        direction = "Negative"
        interpretation = f"As {series1_name} increases, {series2_name} tends to decrease."
    
    summary += f"Correlation Direction: {direction}\n"
    summary += f"Interpretation: {interpretation}\n"
    
    # Add significance note
    if metrics['pearson_p'] < 0.05:
        summary += "\nNote: Correlation is statistically significant (p < 0.05)."
    else:
        summary += "\nNote: Correlation is not statistically significant (p ≥ 0.05)."
    
    return summary 