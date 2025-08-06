#!/usr/bin/env python3
"""
Synchronized Video and Biomechanical Data Visualization (Qt Version)

This script displays video together with biomechanical data (insole force, QTM force,
OpenCap and QTM joint angles) in synchronized plots with a cursor.

This version uses a modern Qt-based interface for better usability.
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtWidgets

from sync_video.config import (
    DEFAULT_VIDEO_PATH,
    DEFAULT_OPENCAP_FILE,
    DEFAULT_QTM_FILE,
    DEFAULT_INSOLE_FILE,
    DEFAULT_INSOLE_OFFSET,
    DEFAULT_QTM_FORCE_OFFSET,
    DEFAULT_OPENCAP_KNEE_OFFSET,
    DEFAULT_QTM_KNEE_OFFSET
)
from sync_video.loaders import load_opencap_data, load_qtm_data, load_insole_data
from sync_video.processors import (
    extract_opencap_joint_angles,
    extract_qtm_joint_angles,
    extract_qtm_force_data
)
from sync_video.gui.qt_player import QtSynchronizedVideoPlayer

import argparse


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qt-based synchronized visualization of video and biomechanical data"
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_PATH,
        help="Path to video file"
    )
    parser.add_argument(
        "--opencap",
        default=DEFAULT_OPENCAP_FILE,
        help="Path to OpenCap motion file (.mot)"
    )
    parser.add_argument(
        "--qtm",
        default=DEFAULT_QTM_FILE,
        help="Path to QTM data file (.csv)"
    )
    parser.add_argument(
        "--insole",
        default=DEFAULT_INSOLE_FILE,
        help="Path to insole data file (.csv)"
    )
    parser.add_argument(
        "--insole-offset",
        type=float,
        default=DEFAULT_INSOLE_OFFSET,
        help="Time offset for insole data (seconds)"
    )
    parser.add_argument(
        "--qtm-force-offset",
        type=float,
        default=DEFAULT_QTM_FORCE_OFFSET,
        help="Time offset for QTM force data (seconds)"
    )
    parser.add_argument(
        "--opencap-offset",
        type=float,
        default=DEFAULT_OPENCAP_KNEE_OFFSET,
        help="Time offset for OpenCap data (seconds)"
    )
    parser.add_argument(
        "--qtm-knee-offset",
        type=float,
        default=DEFAULT_QTM_KNEE_OFFSET,
        help="Time offset for QTM knee data (seconds)"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Load data
    print("Loading data files...")
    opencap_data = load_opencap_data(args.opencap)
    qtm_data = load_qtm_data(args.qtm)
    insole_data = load_insole_data(args.insole)

    # Extract specific data
    print("Processing data...")
    opencap_joint_data = None
    if opencap_data is not None:
        opencap_joint_data = extract_opencap_joint_angles(opencap_data)
    qtm_force_data = extract_qtm_force_data(qtm_data)
    qtm_joint_data = extract_qtm_joint_angles(qtm_data)

    # Create the synchronized video player
    print("Initializing player...")
    player = QtSynchronizedVideoPlayer(
        args.video, 
        insole_data, 
        qtm_force_data, 
        opencap_joint_data, 
        qtm_joint_data,
        insole_offset=args.insole_offset,
        qtm_force_offset=args.qtm_force_offset,
        opencap_knee_offset=args.opencap_offset,
        qtm_knee_offset=args.qtm_knee_offset
    )

    # Show player instructions
    print("\nQt Video Player Controls:")
    print("- Space: Play/Pause")
    print("- Left/Right Arrow: Previous/Next Frame")
    print("- Slider: Seek through video")
    print("- Dropdown menus: Select parameters to display")
    
    # Show the player window
    player.showMaximized()
    
    # Start the Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()