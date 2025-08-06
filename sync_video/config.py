"""
Configuration constants for the sync_video application
"""

import os

# Paths to data files
DEFAULT_VIDEO_PATH = "/Volumes/T7Shield/TAU/biomechanical/sync-recording/data/opencap_data/OpenCapData_eec9c938-f468-478d-aab1-4ffcb0963207_1507-3/Videos/Cam1/InputMedia/1507-3/1507-3_sync.mp4"
DATA_DIR = "/Volumes/T7Shield/TAU/biomechanical/sync-recording/data"
DEFAULT_OPENCAP_FILE = os.path.join(
    DATA_DIR,
    "opencap_data/OpenCapData_eec9c938-f468-478d-aab1-4ffcb0963207_1507-3/OpenSimData/Kinematics/1507-3.mot",
)
DEFAULT_QTM_FILE = os.path.join(DATA_DIR, "qtm_data/qtm_data_20250324_182608.csv")
DEFAULT_INSOLE_FILE = os.path.join(DATA_DIR, "insole_data/insole_data_20250324_182608.csv")

# Data offset defaults (in seconds)
DEFAULT_INSOLE_OFFSET = 3.15
DEFAULT_QTM_FORCE_OFFSET = 3.31
DEFAULT_OPENCAP_KNEE_OFFSET = 0.0
DEFAULT_QTM_KNEE_OFFSET = 3.31