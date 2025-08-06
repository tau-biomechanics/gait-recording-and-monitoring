import os
import tempfile
import numpy as np
from pathlib import Path

def create_sample_trc_file(output_file='sample.trc'):
    """
    Create a sample TRC file with the format that OpenSim expects.
    """
    # Create some sample marker data
    n_frames = 10
    frames = np.arange(1, n_frames + 1)
    times = np.linspace(0, 1, n_frames)
    
    # Create 3 markers
    markers = ['marker1', 'marker2', 'marker3']
    
    # Create sample marker data (random positions)
    markers_data = np.random.rand(n_frames, 3 * len(markers))
    
    # Write the TRC file
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_file}\n")
        frame_rate = 100.0
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{frame_rate:.1f}\t{frame_rate:.1f}\t{n_frames}\t{len(markers)}\tmm\t{frame_rate:.1f}\t1\t{n_frames}\n")
        
        # Write marker names on one line
        marker_line = "Frame#\tTime"
        for marker in markers:
            marker_line += f"\t{marker}"
        f.write(marker_line + "\n")
        
        # Write component labels on the next line
        component_line = "\t"
        for i in range(len(markers)):
            component_line += f"\tX{i+1}\tY{i+1}\tZ{i+1}"
        f.write(component_line + "\n")
        
        # Write the actual data
        for i in range(n_frames):
            data_line = f"{int(frames[i])}\t{times[i]:.6f}"
            for j in range(len(markers)):
                x = markers_data[i, j*3]
                y = markers_data[i, j*3+1]
                z = markers_data[i, j*3+2]
                data_line += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
            f.write(data_line + "\n")
    
    print(f"Created sample TRC file: {output_file}")
    
    # Print the contents of the file
    print("\nFile contents:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f, 1):
            print(f"{i}: {line.strip()}")

if __name__ == "__main__":
    create_sample_trc_file() 