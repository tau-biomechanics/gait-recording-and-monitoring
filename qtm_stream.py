#!/usr/bin/env python3
"""
QTM Real-time Streaming with Moticon Insole Integration
This script listens for UDP packets from Moticon insoles and triggers QTM recording.
Data from QTM and insoles are stored in separate files.
"""

import asyncio
import argparse
import datetime
import qtm_rt
import csv
import os
import signal
import socket
import threading
import traceback

# Global variables
running = True
qtm_output_file = None
insole_output_file = None
qtm_csv_writer = None
insole_csv_writer = None
qtm_file_handle = None
insole_file_handle = None
qtm_connection = None
qtm_streaming = False
insole_headers = None  # Will store the insole column headers


def signal_handler(sig, frame):
    """Handle CTRL+C to cleanly exit the program"""
    global running
    print("\nShutting down...")
    running = False

    # Force exit if program is stuck
    print("Forcing exit in 3 seconds if program doesn't terminate...")
    # Schedule a force exit after a delay
    timer = threading.Timer(3.0, lambda: os._exit(0))
    timer.daemon = True  # So it doesn't block program exit
    timer.start()

    # Cancel all running tasks
    for task in asyncio.all_tasks():
        task.cancel()


async def setup_connection(host, port=22223):
    """Connect to QTM"""
    print(f"Connecting to QTM at {host}:{port}...")
    try:
        connection = await qtm_rt.connect(host, port)

        if connection is None:
            raise RuntimeError("Failed to connect to QTM")

        # Get QTM info without using version method
        print(f"Connected to QTM")

        # Check if QTM is in the right state
        info = await connection.get_state()
        print(f"QTM State: {info}")

        try:
            # Try to get marker labels
            xml_string = await connection.get_parameters(parameters=["3d"])
            print("Retrieved 3D parameters from QTM")
            # Print the first 200 characters to avoid flooding the console
            print(f"Parameters: {xml_string[:200]}...")
        except Exception as e:
            print(f"Could not get marker labels: {e}")

        return connection
    except Exception as e:
        print(f"Error connecting to QTM: {e}")
        raise


def create_output_files(output_dir):
    """Create separate CSV files for QTM and insole data"""
    global \
        qtm_output_file, \
        insole_output_file, \
        qtm_csv_writer, \
        insole_csv_writer, \
        qtm_file_handle, \
        insole_file_handle, \
        insole_headers

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create QTM data file
    qtm_output_file = os.path.join(output_dir, f"qtm_data_{timestamp}.csv")
    qtm_file_handle = open(qtm_output_file, "w", newline="")
    qtm_csv_writer = csv.writer(qtm_file_handle)

    # Create insole data file
    insole_output_file = os.path.join(output_dir, f"insole_data_{timestamp}.csv")
    insole_file_handle = open(insole_output_file, "w", newline="")
    insole_csv_writer = csv.writer(insole_file_handle)

    # Write header for insole file
    if insole_headers is None:
        # Default headers if none provided
        insole_headers = [
            "Timestamp",
            "Left_Total_Force",
            "Right_Total_Force",
            "Left_COP_X",
            "Left_COP_Y",
            "Right_COP_X",
            "Right_COP_Y",
            "Stance_Phase",
            "Gait_Line",
        ]

    insole_csv_writer.writerow(insole_headers)

    print(f"Saving QTM data to: {qtm_output_file}")
    print(f"Saving insole data to: {insole_output_file}")
    return qtm_csv_writer, insole_csv_writer


def on_packet(packet):
    """Callback for handling incoming QTM packets"""
    global running, qtm_csv_writer

    if not running:
        return

    try:
        # Get basic packet info
        try:
            frame_number = packet.framenumber
            timestamp = packet.timestamp
        except AttributeError:
            print("Warning: Received QTM packet without basic attributes")
            return

        row = [frame_number, timestamp]

        # Print debug info about packet components once
        if not hasattr(on_packet, "component_debug_shown") and hasattr(
            packet, "components"
        ):
            print("Available packet components:")
            for comp in packet.components:
                print(f"  - {comp}")
            on_packet.component_debug_shown = True

            # Print the full list of available component types for reference
            print("All possible component types:")
            for comp_type in dir(qtm_rt.packet.QRTComponentType):
                if not comp_type.startswith("_"):
                    print(f"  - {comp_type}")

        # Process 3D data if available
        markers_data = []
        try:
            # First check if packet has components attribute
            if hasattr(packet, "components") and packet.components is not None:
                if qtm_rt.packet.QRTComponentType.Component3d in packet.components:
                    try:
                        header, markers = packet.get_3d_markers()

                        # Print debug info on first packet
                        if not hasattr(on_packet, "marker_debug_shown"):
                            print(f"3D marker header info: {header}")
                            print(f"Number of markers: {len(markers)}")
                            if markers:
                                first_marker = markers[0]
                                print(f"First marker type: {type(first_marker)}")
                                print(f"First marker attributes: {dir(first_marker)}")
                                print(f"First marker sample data:")
                                print(
                                    f"  - x: {first_marker.x} (type: {type(first_marker.x)})"
                                )
                                print(
                                    f"  - y: {first_marker.y} (type: {type(first_marker.y)})"
                                )
                                print(
                                    f"  - z: {first_marker.z} (type: {type(first_marker.z)})"
                                )
                            on_packet.marker_debug_shown = True

                        # Store marker data
                        for marker in markers:
                            try:
                                # Handle occluded markers - set to explicit zeros rather than None
                                x = 0.0
                                y = 0.0
                                z = 0.0

                                if (
                                    marker.x is not None
                                    and marker.y is not None
                                    and marker.z is not None
                                ):
                                    try:
                                        x = float(marker.x)
                                        y = float(marker.y)
                                        z = float(marker.z)
                                    except (TypeError, ValueError) as err:
                                        print(
                                            f"Error converting marker coordinates to float: {err}"
                                        )

                                # Add marker data to the row
                                markers_data.extend([x, y, z])
                            except Exception as marker_err:
                                print(
                                    f"Error processing individual marker: {marker_err}"
                                )
                                # Add zeros for this marker
                                markers_data.extend([0.0, 0.0, 0.0])
                    except Exception as e:
                        print(f"Error processing 3D markers: {e}")
                        import traceback

                        traceback_info = traceback.format_exc()
                        print(f"Marker processing traceback: {traceback_info}")
        except Exception as e:
            print(f"Error processing component data: {e}")

        # Process 6D (joint angle) data if available
        joint_data = []
        try:
            if hasattr(packet, "components") and packet.components is not None:
                if qtm_rt.packet.QRTComponentType.Component6d in packet.components:
                    try:
                        header, bodies = packet.get_6d()

                        # Print debug info on first packet
                        if not hasattr(on_packet, "joint_debug_shown"):
                            print(f"6D data header info: {header}")
                            print(f"Number of rigid bodies: {len(bodies)}")
                            if bodies:
                                first_body = bodies[0]
                                print(f"First body type: {type(first_body)}")
                                print(f"First body attributes: {dir(first_body)}")
                                print(f"First body sample data:")
                                print(
                                    f"  - x: {first_body.x} (type: {type(first_body.x) if first_body.x is not None else None})"
                                )
                                print(
                                    f"  - y: {first_body.y} (type: {type(first_body.y) if first_body.y is not None else None})"
                                )
                                print(
                                    f"  - z: {first_body.z} (type: {type(first_body.z) if first_body.z is not None else None})"
                                )
                                print(
                                    f"  - rotation: {first_body.rotation} (type: {type(first_body.rotation) if first_body.rotation is not None else None})"
                                )
                            on_packet.joint_debug_shown = True

                        # Store 6D (joint) data
                        for body in bodies:
                            try:
                                # Position - set defaults to 0
                                x = 0.0
                                y = 0.0
                                z = 0.0

                                # Set rotation defaults
                                rw = 0.0
                                rx = 0.0
                                ry = 0.0
                                rz = 0.0

                                # Extract position if available
                                if (
                                    body.x is not None
                                    and body.y is not None
                                    and body.z is not None
                                ):
                                    try:
                                        x = float(body.x)
                                        y = float(body.y)
                                        z = float(body.z)
                                    except (TypeError, ValueError) as err:
                                        print(
                                            f"Error converting body position to float: {err}"
                                        )

                                # Extract rotation if available
                                if (
                                    body.rotation is not None
                                    and len(body.rotation) >= 4
                                ):
                                    try:
                                        rw = float(body.rotation[0])
                                        rx = float(body.rotation[1])
                                        ry = float(body.rotation[2])
                                        rz = float(body.rotation[3])
                                    except (TypeError, ValueError) as err:
                                        print(
                                            f"Error converting body rotation to float: {err}"
                                        )

                                # Add position and rotation to data
                                joint_data.extend([x, y, z, rw, rx, ry, rz])
                            except Exception as body_err:
                                print(
                                    f"Error processing individual rigid body: {body_err}"
                                )
                                # Add zeros for this body
                                joint_data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    except Exception as e:
                        print(f"Error processing 6D data: {e}")
                        import traceback

                        traceback_info = traceback.format_exc()
                        print(f"6D data processing traceback: {traceback_info}")
        except Exception as e:
            print(f"Error checking for 6D components: {e}")

        # Process force data if available
        force_data = []
        try:
            # First check if packet has components attribute
            if hasattr(packet, "components") and packet.components is not None:
                if qtm_rt.packet.QRTComponentType.ComponentForce in packet.components:
                    try:
                        # Get force data from the packet
                        force_data_result = packet.get_force()

                        # Only log the structure once for debugging
                        if not hasattr(on_packet, "force_debug_shown"):
                            print(f"Force data type: {type(force_data_result)}")
                            # Extract useful info for debugging
                            if (
                                isinstance(force_data_result, tuple)
                                and len(force_data_result) >= 2
                            ):
                                force_header, force_plates = force_data_result
                                print(f"Force plates count: {len(force_plates)}")
                                if len(force_plates) > 0:
                                    # Print the type and first few attributes of a force plate
                                    first_plate = force_plates[0]
                                    print(f"Force plate type: {type(first_plate)}")
                                    print(
                                        f"Force plate repr: {repr(first_plate)[:100]}..."
                                    )
                            on_packet.force_debug_shown = True

                        # Process the force data
                        if (
                            isinstance(force_data_result, tuple)
                            and len(force_data_result) >= 2
                        ):
                            # Get the list of force plates
                            _, force_plates = force_data_result

                            # Process each force plate's data
                            for force_plate in force_plates:
                                # Check if it's a "RTForcePlate" object by examining its string representation
                                force_plate_str = str(force_plate)

                                # If it's an RTForcePlate object with RTForce objects inside
                                if "RTForcePlate" in force_plate_str:
                                    # Try to extract force data from the forces attribute if it exists
                                    if (
                                        hasattr(force_plate, "forces")
                                        and force_plate.forces
                                        and len(force_plate.forces) > 0
                                    ):
                                        # Take the first force measurement
                                        first_force = force_plate.forces[0]
                                        # Extract components
                                        if hasattr(first_force, "x"):
                                            force_data.extend(
                                                [
                                                    first_force.x,
                                                    first_force.y,
                                                    first_force.z,
                                                ]
                                            )
                                            force_data.extend(
                                                [
                                                    first_force.x_m,
                                                    first_force.y_m,
                                                    first_force.z_m,
                                                ]
                                            )
                                            force_data.extend(
                                                [
                                                    first_force.x_a,
                                                    first_force.y_a,
                                                    first_force.z_a,
                                                ]
                                            )
                                        else:
                                            # Fallback to zeros
                                            force_data.extend(
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0]
                                            )
                                    else:
                                        # Try to parse forces from the string representation
                                        # Format looks like: RTForcePlate(id=2, force_count=17, force_number=1491)","[RTForce(x=3.87, y=-4.59...
                                        try:
                                            # Look for the RTForce text in the string
                                            if "RTForce" in force_plate_str:
                                                # Extract from string representation - using the first force object
                                                import re

                                                # Extract x, y, z values
                                                force_xyz = re.search(
                                                    r"RTForce\(x=([+-]?\d+\.\d+), y=([+-]?\d+\.\d+), z=([+-]?\d+\.\d+)",
                                                    force_plate_str,
                                                )
                                                if force_xyz:
                                                    fx = float(force_xyz.group(1))
                                                    fy = float(force_xyz.group(2))
                                                    fz = float(force_xyz.group(3))
                                                    force_data.extend([fx, fy, fz])
                                                else:
                                                    force_data.extend([0, 0, 0])

                                                # Extract x_m, y_m, z_m (moment) values
                                                moment_xyz = re.search(
                                                    r"x_m=([+-]?\d+\.\d+), y_m=([+-]?\d+\.\d+), z_m=([+-]?\d+\.\d+)",
                                                    force_plate_str,
                                                )
                                                if moment_xyz:
                                                    mx = float(moment_xyz.group(1))
                                                    my = float(moment_xyz.group(2))
                                                    mz = float(moment_xyz.group(3))
                                                    force_data.extend([mx, my, mz])
                                                else:
                                                    force_data.extend([0, 0, 0])

                                                # Extract x_a, y_a, z_a (COP) values
                                                cop_xyz = re.search(
                                                    r"x_a=([+-]?\d+\.\d+), y_a=([+-]?\d+\.\d+), z_a=([+-]?\d+\.\d+)",
                                                    force_plate_str,
                                                )
                                                if cop_xyz:
                                                    cx = float(cop_xyz.group(1))
                                                    cy = float(cop_xyz.group(2))
                                                    cz = float(cop_xyz.group(3))
                                                    force_data.extend([cx, cy, cz])
                                                else:
                                                    force_data.extend([0, 0, 0])
                                            else:
                                                # No RTForce found in string
                                                force_data.extend(
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]
                                                )
                                        except Exception as parse_err:
                                            print(
                                                f"Error parsing force data from string: {parse_err}"
                                            )
                                            force_data.extend(
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0]
                                            )
                                else:
                                    # Try the previous approach with tuples
                                    if (
                                        isinstance(force_plate, tuple)
                                        and len(force_plate) >= 9
                                    ):
                                        force_data.extend(force_plate[:9])
                                    else:
                                        # Add zeros as placeholder
                                        force_data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
                    except Exception as e:
                        print(f"Error processing force data: {e}")
                        import traceback

                        traceback_info = traceback.format_exc()
                        print(f"Traceback: {traceback_info}")
                        # Add zeros for this force plate
                        force_data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        except Exception as e:
            print(f"Error checking for force components: {e}")

        # First packet: write header for QTM data
        if not hasattr(on_packet, "first_packet"):
            try:
                # Create header row
                header_row = ["Frame", "Timestamp"]

                # Add marker headers if we have marker data
                if (
                    markers_data
                    and hasattr(packet, "components")
                    and packet.components is not None
                ):
                    if qtm_rt.packet.QRTComponentType.Component3d in packet.components:
                        header, markers = packet.get_3d_markers()
                        for i, marker in enumerate(markers):
                            header_row.extend(
                                [
                                    f"Marker{i + 1}_X",
                                    f"Marker{i + 1}_Y",
                                    f"Marker{i + 1}_Z",
                                ]
                            )

                # Add joint angle headers if we have 6D data
                if (
                    joint_data
                    and hasattr(packet, "components")
                    and packet.components is not None
                ):
                    if qtm_rt.packet.QRTComponentType.Component6d in packet.components:
                        header, bodies = packet.get_6d()
                        for i, body in enumerate(bodies):
                            header_row.extend(
                                [
                                    f"Joint{i + 1}_X",
                                    f"Joint{i + 1}_Y",
                                    f"Joint{i + 1}_Z",
                                    f"Joint{i + 1}_QW",
                                    f"Joint{i + 1}_QX",
                                    f"Joint{i + 1}_QY",
                                    f"Joint{i + 1}_QZ",
                                ]
                            )

                # Add force plate headers if we have force data
                if hasattr(packet, "components") and packet.components is not None:
                    if (
                        qtm_rt.packet.QRTComponentType.ComponentForce
                        in packet.components
                    ):
                        try:
                            # Get force data to determine number of force plates
                            force_data_result = packet.get_force()

                            if (
                                isinstance(force_data_result, tuple)
                                and len(force_data_result) >= 2
                            ):
                                # Get the list of force plates
                                _, force_plates = force_data_result

                                # Count how many real force plates there are
                                num_force_plates = len(force_plates)

                                # Generate one set of headers for each force plate
                                for i in range(num_force_plates):
                                    # Force vectors (x, y, z for each force plate)
                                    header_row.extend(
                                        [
                                            f"ForcePlate{i + 1}_ForceX",
                                            f"ForcePlate{i + 1}_ForceY",
                                            f"ForcePlate{i + 1}_ForceZ",
                                        ]
                                    )

                                    # Moment vectors (x, y, z for each force plate)
                                    header_row.extend(
                                        [
                                            f"ForcePlate{i + 1}_MomentX",
                                            f"ForcePlate{i + 1}_MomentY",
                                            f"ForcePlate{i + 1}_MomentZ",
                                        ]
                                    )

                                    # Center of pressure (x, y, z for each force plate)
                                    header_row.extend(
                                        [
                                            f"ForcePlate{i + 1}_CoPX",
                                            f"ForcePlate{i + 1}_CoPY",
                                            f"ForcePlate{i + 1}_CoPZ",
                                        ]
                                    )

                                print(
                                    f"Generated headers for {num_force_plates} force plates"
                                )
                        except Exception as e:
                            print(f"Error generating force plate headers: {e}")
                            import traceback

                            traceback_info = traceback.format_exc()
                            print(f"Header generation traceback: {traceback_info}")

                            # Fallback to default headers
                            print("Falling back to default force plate headers")
                            header_row.extend(
                                [
                                    "ForcePlate1_ForceX",
                                    "ForcePlate1_ForceY",
                                    "ForcePlate1_ForceZ",
                                    "ForcePlate1_MomentX",
                                    "ForcePlate1_MomentY",
                                    "ForcePlate1_MomentZ",
                                    "ForcePlate1_CoPX",
                                    "ForcePlate1_CoPY",
                                    "ForcePlate1_CoPZ",
                                ]
                            )

                qtm_csv_writer.writerow(header_row)
                on_packet.first_packet = True
                print(f"Started writing QTM data with header: {header_row}")
            except Exception as e:
                print(f"Error creating CSV header: {e}")
                traceback_info = traceback.format_exc()
                print(f"Header generation traceback: {traceback_info}")
                on_packet.first_packet = True  # Mark as processed even if failed

        # Combine QTM data and write row
        row.extend(markers_data)
        row.extend(joint_data)
        row.extend(force_data)
        qtm_csv_writer.writerow(row)

        # Print status every 100 frames
        if frame_number % 100 == 0:
            print(f"Streaming... Frame: {frame_number}")

    except Exception as e:
        print(f"Error processing QTM packet: {e}")
        # Continue processing packets even if this one failed


async def start_qtm_recording(connection, qtm_password):
    """Take control of QTM and start a new recording"""
    global qtm_streaming

    try:
        print("Starting QTM recording...")

        # First check the current state of QTM
        current_state = await connection.get_state()
        print(f"Current QTM state: {current_state}")

        # Components to stream - use only valid component names
        components_to_stream = ["3d", "6d", "force"]
        print(f"Requesting components: {components_to_stream}")

        # Handle different QTM states
        if current_state == qtm_rt.QRTEvent.EventWaitingForTrigger:
            print("QTM is waiting for trigger, sending trigger now...")

            # Take control to send trigger
            async with qtm_rt.TakeControl(connection, qtm_password):
                # Trigger the measurement to start
                try:
                    await connection.trig()
                    print("Trigger sent successfully")

                    # Wait for capture to start
                    for _ in range(10):  # Try for 1 second
                        await asyncio.sleep(0.1)
                        new_state = await connection.get_state()
                        if new_state == qtm_rt.QRTEvent.EventCaptureStarted:
                            print("QTM capture started after trigger")
                            break

                    # Try to start streaming data with all components
                    result = await connection.stream_frames(
                        components=components_to_stream, on_packet=on_packet
                    )
                    if result:
                        qtm_streaming = True
                        print("Successfully started streaming after trigger")
                        return True
                    else:
                        print(
                            "Failed to stream with all components, trying with minimal components..."
                        )
                        # Try with just 3D and 6D
                        result = await connection.stream_frames(
                            components=["3d", "6d"], on_packet=on_packet
                        )
                        if result:
                            qtm_streaming = True
                            print(
                                "Successfully started streaming with minimal components"
                            )
                            return True
                        else:
                            print("Failed to start streaming with minimal components")
                except Exception as e:
                    print(f"Error sending trigger: {e}")
                    # Continue to other approaches

        # If already in capture state, just attach to the stream
        elif current_state == qtm_rt.QRTEvent.EventCaptureStarted:
            print("QTM is already capturing, attaching to current measurement")
            # Just try to start streaming directly
            try:
                print("Starting direct streaming...")
                result = await connection.stream_frames(
                    components=components_to_stream, on_packet=on_packet
                )
                if result:
                    qtm_streaming = True
                    print("Successfully attached to running measurement")
                    return True
                else:
                    print(
                        "Failed to stream with all components, trying with minimal components..."
                    )
                    # Try with just 3D and 6D
                    result = await connection.stream_frames(
                        components=["3d", "6d"], on_packet=on_packet
                    )
                    if result:
                        qtm_streaming = True
                        print("Successfully started streaming with minimal components")
                        return True
                    else:
                        print("Failed to start streaming with minimal components")
            except Exception as e:
                print(f"Error attaching to running measurement: {e}")
                # Continue to try alternative approaches

        # Try to take control and start a new measurement if needed
        try:
            print("Taking control with async context manager...")
            async with qtm_rt.TakeControl(connection, qtm_password):
                # If we're not in a measurement or waiting for trigger, start one
                if current_state not in [
                    qtm_rt.QRTEvent.EventWaitingForTrigger,
                    qtm_rt.QRTEvent.EventCaptureStarted,
                ]:
                    print("Starting new QTM measurement...")
                    result = await connection.new()
                    if not result:
                        print("Failed to start new measurement")
                        return False
                    print("New measurement started")
                else:
                    print("Using existing measurement")

                # Start streaming directly without checking components
                print("Starting QTM streaming...")
                try:
                    # First try with all components
                    result = await connection.stream_frames(
                        components=components_to_stream, on_packet=on_packet
                    )

                    if not result:
                        print(
                            "Failed to stream with all components, trying with minimal components..."
                        )
                        # Try with just 3D and 6D
                        result = await connection.stream_frames(
                            components=["3d", "6d"], on_packet=on_packet
                        )
                        if result:
                            qtm_streaming = True
                            print(
                                "Successfully started streaming with minimal components"
                            )
                            return True
                        else:
                            print(
                                "Failed to stream with default components, trying with no specific components..."
                            )
                            # Try with default components
                            result = await connection.stream_frames(on_packet=on_packet)

                            if not result:
                                print(
                                    "Failed to start streaming with default components"
                                )
                                return False
                except Exception as e:
                    print(f"QTM streaming error: {e}")
                    print("Trying alternative streaming approach...")

                    # Alternative approach: just start the current measurement
                    try:
                        # Only start if not already started
                        if current_state == qtm_rt.QRTEvent.EventWaitingForTrigger:
                            # Try to trigger the measurement
                            await connection.trig()
                            print("Triggered QTM measurement")
                        elif current_state != qtm_rt.QRTEvent.EventCaptureStarted:
                            await connection.start()
                            print("Started QTM measurement directly")
                    except Exception as start_error:
                        if b"Measurement is already running" in str(start_error):
                            print("Measurement already running, continuing...")
                        else:
                            print(f"Error starting measurement: {start_error}")

                qtm_streaming = True
                print("QTM recording and streaming started")
                return True

        except Exception as e:
            print(f"Error during QTM operation: {e}")
            # If the error is about measurement already running, consider it a success
            if b"Measurement is already running" in str(e):
                print("Measurement already running, continuing...")
                qtm_streaming = True
                return True
            if hasattr(e, "__dict__"):
                print(f"Error details: {e.__dict__}")
            return False

    except Exception as e:
        print(f"Failed to start QTM recording: {e}")
        return False


async def trigger_qtm(connection, password):
    """Send a trigger to QTM"""
    try:
        async with qtm_rt.TakeControl(connection, password):
            # According to QTM documentation, trig() is the correct method to send a trigger
            result = await connection.trig()
            print(f"Trigger sent to QTM: {result}")
            return result
    except Exception as e:
        print(f"Error sending trigger to QTM: {e}")
        return False


async def udp_listener(
    ip, port, qtm_host, qtm_port, qtm_password, output_dir, immediate_trigger=False
):
    """Listen for UDP packets from Moticon insoles and trigger QTM recording"""
    global running, qtm_connection, qtm_streaming, insole_csv_writer

    print(f"Starting UDP listener on {ip}:{port}")
    print("Waiting for first Moticon insole data to trigger QTM recording...")

    # Components to stream - use only valid component names
    components_to_stream = ["3d", "6d", "force"]

    # Create a UDP socket - use "0.0.0.0" to listen on all interfaces if needed
    listen_ip = "0.0.0.0" if ip == "127.0.0.1" else ip
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((listen_ip, port))
        print(f"Successfully bound to {listen_ip}:{port}")
    except Exception as e:
        print(f"Error binding to {listen_ip}:{port}: {e}")
        running = False
        return

    # Connect to QTM first but don't start recording yet
    try:
        qtm_connection = await setup_connection(qtm_host, qtm_port)
        if qtm_connection is None:
            print("Failed to connect to QTM. Exiting.")
            running = False
            return

        # Check if QTM is already in a ready state
        current_state = await qtm_connection.get_state()
        print(f"Initial QTM state: {current_state}")

        if current_state == qtm_rt.QRTEvent.EventCaptureStarted:
            print("QTM is already recording, marking as streaming")
            qtm_streaming = True
        elif current_state == qtm_rt.QRTEvent.EventWaitingForTrigger:
            print("QTM is waiting for trigger")

            # If immediate trigger is requested, trigger QTM now
            if immediate_trigger:
                print("Immediate trigger requested. Sending trigger now...")
                trigger_result = await trigger_qtm(qtm_connection, qtm_password)
                if trigger_result:
                    # Wait for QTM to start
                    for _ in range(10):  # Try for 1 second
                        await asyncio.sleep(0.1)
                        new_state = await qtm_connection.get_state()
                        if new_state == qtm_rt.QRTEvent.EventCaptureStarted:
                            print("QTM capture started after immediate trigger")
                            break

                    # Start streaming data
                    try:
                        print(
                            f"Starting streaming with components: {components_to_stream}"
                        )
                        result = await qtm_connection.stream_frames(
                            components=components_to_stream, on_packet=on_packet
                        )
                        if result:
                            qtm_streaming = True
                            print(
                                "Successfully started streaming after immediate trigger"
                            )
                        else:
                            print(
                                "Failed to stream with all components, trying with minimal components..."
                            )
                            # Try with just 3D and 6D
                            result = await qtm_connection.stream_frames(
                                components=["3d", "6d"], on_packet=on_packet
                            )
                            if result:
                                qtm_streaming = True
                                print(
                                    "Successfully started streaming with minimal components"
                                )
                            else:
                                print("Failed to stream data after immediate trigger")
                    except Exception as e:
                        print(f"Error starting stream after immediate trigger: {e}")
            else:
                print("Will trigger on first insole packet")

    except Exception as e:
        print(f"QTM connection failed: {e}")
        running = False
        return

    # Create output files
    create_output_files(output_dir)

    # Main loop to receive UDP packets
    first_packet_received = False
    sock.settimeout(0.5)  # 500ms timeout

    async def check_for_packets():
        nonlocal first_packet_received
        global qtm_streaming
        initial_qtm_state = await qtm_connection.get_state()

        while running:
            try:
                try:
                    data, addr = sock.recvfrom(2048)
                    print(f"Received UDP packet from {addr[0]}:{addr[1]}")

                    # Parse the UDP packet
                    data_str = data.decode("utf-8").strip()
                    values = data_str.split()
                    values = [float(v) for v in values]

                    if values:
                        # Use the timestamp from the insole data
                        timestamp = values[0]

                        # Write insole data directly to its own file
                        insole_csv_writer.writerow(values)

                        # If this is the first packet, handle QTM based on its state
                        if not first_packet_received:
                            first_packet_received = True
                            print("First Moticon insole data received.")

                            # If QTM was waiting for trigger when we started, directly trigger it
                            if (
                                initial_qtm_state
                                == qtm_rt.QRTEvent.EventWaitingForTrigger
                                and not qtm_streaming
                            ):
                                print(
                                    "QTM was waiting for trigger. Sending trigger now..."
                                )
                                await trigger_qtm(qtm_connection, qtm_password)

                                # Wait a bit for QTM to start capturing
                                await asyncio.sleep(0.5)

                                # Now start streaming
                                try:
                                    print(
                                        f"Starting streaming with components: {components_to_stream}"
                                    )
                                    result = await qtm_connection.stream_frames(
                                        components=components_to_stream,
                                        on_packet=on_packet,
                                    )
                                    if result:
                                        qtm_streaming = True
                                        print(
                                            "Successfully started streaming after trigger"
                                        )
                                    else:
                                        print(
                                            "Failed to stream with all components, trying with minimal components..."
                                        )
                                        # Try with just 3D and 6D
                                        result = await qtm_connection.stream_frames(
                                            components=["3d", "6d"], on_packet=on_packet
                                        )
                                        if result:
                                            qtm_streaming = True
                                            print(
                                                "Successfully started streaming with minimal components"
                                            )
                                        else:
                                            print("Failed to stream data after trigger")
                                except Exception as e:
                                    print(f"Error starting stream after trigger: {e}")

                            # Otherwise use the regular approach
                            elif not qtm_streaming:
                                print("Starting QTM recording...")
                                await start_qtm_recording(qtm_connection, qtm_password)

                except socket.timeout:
                    # No data available, wait a bit
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error receiving UDP data: {e}")
                    await asyncio.sleep(0.1)

                # Yield control back to event loop occasionally
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                print("UDP listener task cancelled")
                break
            except Exception as e:
                print(f"Unexpected error in UDP listener: {e}")
                await asyncio.sleep(0.1)

    # Start the packet checking task
    packet_task = asyncio.create_task(check_for_packets())

    # Wait for the task to complete or be cancelled
    try:
        await packet_task
    except asyncio.CancelledError:
        packet_task.cancel()
        try:
            await packet_task
        except asyncio.CancelledError:
            pass
    finally:
        sock.close()
        print("UDP listener stopped")


async def cleanup():
    """Clean up resources"""
    global \
        qtm_file_handle, \
        insole_file_handle, \
        qtm_connection, \
        qtm_streaming, \
        qtm_output_file, \
        insole_output_file

    if qtm_connection and qtm_streaming:
        try:
            print("Stopping QTM streaming...")
            try:
                # Stop streaming
                await qtm_connection.stream_frames_stop()
                print("Streaming stopped")

                # Take control again to properly stop the measurement
                async with qtm_rt.TakeControl(qtm_connection, ""):
                    # Stop the current measurement
                    await qtm_connection.stop()
                    print("Stopped QTM measurement")
            except Exception as e:
                print(f"Error stopping QTM streaming: {e}")

            try:
                # Close connection
                await qtm_connection.close()
                print("Closed QTM connection")
            except Exception as e:
                print(f"Error closing QTM connection: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    # Close QTM file
    if qtm_file_handle and not qtm_file_handle.closed:
        qtm_file_handle.close()
        print(f"QTM data saved to {qtm_output_file}")

    # Close insole file
    if insole_file_handle and not insole_file_handle.closed:
        insole_file_handle.close()
        print(f"Insole data saved to {insole_output_file}")


def check_environment():
    """Check if the required modules are available and print version information"""
    print("Checking environment...")
    try:
        # Try to get version if available
        version = getattr(qtm_rt, "__version__", "Version unknown")
        print(f"QTM RT SDK version: {version}")
        print("QTM RT SDK is available")

        # Check available event constants to verify the module is loaded correctly
        print(f"Available QTM events: {dir(qtm_rt.QRTEvent)}")

        return True
    except Exception as e:
        print(f"Error checking QTM environment: {e}")
        return False


async def main(args):
    """Main function"""
    global running, insole_headers

    # Register signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    # Set insole headers if provided
    if args.insole_headers:
        insole_headers = args.insole_headers
        print(f"Using user-provided insole headers: {insole_headers}")

    # Check environment first
    if not args.test_mode and not check_environment():
        print("Environment check failed. Exiting.")
        return

    try:
        # Start UDP listener which will also trigger QTM when data arrives
        udp_task = asyncio.create_task(
            udp_listener(
                args.insole_ip,
                args.insole_port,
                args.qtm_host,
                args.qtm_port,
                args.qtm_password,
                args.output_dir,
                args.immediate_trigger,
            )
        )

        # Keep the script running until CTRL+C or task completion
        try:
            await udp_task
        except asyncio.CancelledError:
            print("Main task cancelled")

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Clean up
        await cleanup()
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Listen for Moticon insole data and trigger QTM recording"
    )
    parser.add_argument(
        "--qtm-host",
        default="127.0.0.1",
        help="QTM server address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--qtm-port", type=int, default=22223, help="QTM server port (default: 22223)"
    )
    parser.add_argument(
        "--qtm-password",
        default="",
        help="QTM password for taking control (default: '')",
    )
    parser.add_argument(
        "--insole-ip",
        default="0.0.0.0",
        help="IP to listen for insole UDP data (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--insole-port",
        type=int,
        default=5555,
        help="Port to listen for insole UDP data (default: 5555)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for saving output files (default: 'data')",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode without connecting to QTM",
    )
    parser.add_argument(
        "--immediate-trigger",
        action="store_true",
        help="Immediately trigger QTM on startup if it's waiting for trigger",
    )
    parser.add_argument(
        "--insole-headers",
        nargs="+",
        help="Headers for insole data columns (default: Timestamp, Left_Total_Force, Right_Total_Force, etc.)",
    )

    args = parser.parse_args()

    # If in test mode, modify the udp_listener behavior
    if args.test_mode:
        # Override udp_listener to only listen for packets without QTM connection
        async def test_udp_listener(ip, port, *args, immediate_trigger=False, **kwargs):
            """Test UDP listener that doesn't connect to QTM"""
            global running

            print(f"TEST MODE: Starting UDP listener on {ip}:{port}")

            # Create a UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((ip, port))
            sock.settimeout(0.5)

            print(f"TEST MODE: Successfully bound to {ip}:{port}")
            print(f"TEST MODE: Waiting for Moticon insole data packets...")

            packet_count = 0
            start_time = datetime.datetime.now()

            # Main loop to receive UDP packets
            while running:
                try:
                    try:
                        data, addr = sock.recvfrom(2048)
                        packet_count += 1
                        current_time = datetime.datetime.now()
                        elapsed = (current_time - start_time).total_seconds()

                        # Parse the UDP packet
                        data_str = data.decode("utf-8").strip()
                        values = data_str.split()
                        try:
                            values = [float(v) for v in values]
                        except ValueError:
                            pass

                        print(
                            f"[{current_time.strftime('%H:%M:%S.%f')[:-3]}] Packet #{packet_count} from {addr[0]}:{addr[1]}"
                        )
                        print(f"Data: {values}")
                        print(f"Rate: {packet_count / elapsed:.2f} packets/second")
                        print("-" * 40)

                    except socket.timeout:
                        # No data available, wait a bit
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Error receiving UDP data: {e}")
                        await asyncio.sleep(0.1)

                    # Yield control back to event loop occasionally
                    await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    break

            sock.close()
            print(
                f"TEST MODE: UDP listener stopped. Received {packet_count} packets over {elapsed:.2f} seconds"
            )

        # Replace the regular udp_listener with our test version
        globals()["udp_listener"] = test_udp_listener

    asyncio.run(main(args))
