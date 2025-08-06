"""
Synchronization utilities for handling data from different sources
"""

import numpy as np
import pandas as pd


def synchronize_data(video_fps, *data_frames):
    """
    Synchronize multiple data frames to a common time base

    Parameters:
    -----------
    video_fps : float
        Frames per second of the video
    *data_frames : list of (DataFrame, str)
        List of tuples containing (dataframe, time_column_name)

    Returns:
    --------
    tuple
        (list of DataFrames with synchronized time values, common_time_array)
    """
    # Find the common time range across all data frames
    min_times = []
    max_times = []

    for df, time_col in data_frames:
        if df is not None and not df.empty:
            min_times.append(df[time_col].min())
            max_times.append(df[time_col].max())

    if not min_times or not max_times:
        print("Error: No valid data frames to synchronize")
        return None, None

    common_min_time = max(min_times)
    common_max_time = min(max_times)

    print(f"Common time range: {common_min_time:.2f}s to {common_max_time:.2f}s")

    # Create a common time base for all data
    # Match to video frames for perfect synchronization
    video_duration = common_max_time - common_min_time
    num_frames = int(video_duration * video_fps)
    common_time = np.linspace(common_min_time, common_max_time, num_frames)

    # Resample/interpolate each data frame to the common time base
    synchronized_frames = []

    for df, time_col in data_frames:
        if df is not None and not df.empty:
            # Create new DataFrame with synchronized time
            sync_df = pd.DataFrame({"time": common_time})

            # Interpolate all columns to match the common time
            for col in df.columns:
                if col != time_col:
                    # Use linear interpolation to map values to new time points
                    sync_df[col] = np.interp(
                        common_time,
                        df[time_col].values,
                        df[col].values,
                        left=np.nan,
                        right=np.nan,
                    )

            synchronized_frames.append(sync_df)
        else:
            synchronized_frames.append(None)

    return synchronized_frames, common_time