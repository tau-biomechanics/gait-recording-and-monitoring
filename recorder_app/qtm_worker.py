from PyQt6.QtCore import QObject, pyqtSignal, QThread
import asyncio
import qtm_rt # Assuming qtm_rt can be used in a non-async top-level manner for setup
import traceback # For detailed error logging
import socket # Added for socket-related exceptions

class QtmWorker(QObject):
    connection_status = pyqtSignal(str) # e.g., "Connected", "Disconnected", "Error: ..."
    qtm_data_packet = pyqtSignal(object) # Emits the received QTM packet
    qtm_event = pyqtSignal(object) # For QTM events like capture started/stopped
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    
    # Signals for data recording
    header_data_ready = pyqtSignal(list) # Emits list of header strings
    data_row_ready = pyqtSignal(list)    # Emits list of data values for a row

    def __init__(self, host, port, password):
        super().__init__()
        self.host = host
        self.port = port
        self.password = password
        self._running = False
        self.qtm_connection = None
        self.loop = None
        self._is_streaming = False
        self._first_packet_processed_for_header = False
        self._current_components = []
        self._is_preview_streaming = False

    async def _on_packet_internal(self, packet):
        if not self._running or not self._is_streaming:
            return

        try:
            frame_number = packet.framenumber
            timestamp = packet.timestamp
            current_row_data = [frame_number, timestamp]
            
            # --- Adapt 3D, 6D, Force data extraction from qtm_stream.py --- 
            markers_data = []
            joint_data = []
            force_data_list = [] # Changed from force_data to avoid conflict

            # Process 3D data
            if hasattr(packet, "components") and packet.components and qtm_rt.packet.QRTComponentType.Component3d in packet.components:
                _, markers = packet.get_3d_markers()
                for marker in markers:
                    x = float(marker.x) if marker.x is not None else 0.0
                    y = float(marker.y) if marker.y is not None else 0.0
                    z = float(marker.z) if marker.z is not None else 0.0
                    markers_data.extend([x, y, z])
            
            # Process 6D (Rigid Body / Joint) data
            if hasattr(packet, "components") and packet.components and qtm_rt.packet.QRTComponentType.Component6d in packet.components:
                _, bodies = packet.get_6d()
                for body in bodies:
                    x = float(body.x) if body.x is not None else 0.0
                    y = float(body.y) if body.y is not None else 0.0
                    z = float(body.z) if body.z is not None else 0.0
                    rot = body.rotation # This is a tuple (rw, rx, ry, rz)
                    rw, rx, ry, rz = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])) if rot and len(rot) == 4 else (0.0, 0.0, 0.0, 0.0)
                    joint_data.extend([x, y, z, rw, rx, ry, rz])

            # Process Force data
            if hasattr(packet, "components") and packet.components and qtm_rt.packet.QRTComponentType.ComponentForce in packet.components:
                force_plates_data = packet.get_force()
                if force_plates_data:
                    _, force_plates = force_plates_data
                    for plate in force_plates:
                        # Adapted from qtm_stream.py, assuming 'forces' attribute with list of measurements
                        # And taking the first measurement sample for simplicity, as in qtm_stream.py
                        if hasattr(plate, 'forces') and plate.forces and len(plate.forces) > 0:
                            sample = plate.forces[0] # Taking the first sample
                            fx = float(sample.x) if hasattr(sample, 'x') and sample.x is not None else 0.0
                            fy = float(sample.y) if hasattr(sample, 'y') and sample.y is not None else 0.0
                            fz = float(sample.z) if hasattr(sample, 'z') and sample.z is not None else 0.0
                            mx = float(sample.x_m) if hasattr(sample, 'x_m') and sample.x_m is not None else 0.0
                            my = float(sample.y_m) if hasattr(sample, 'y_m') and sample.y_m is not None else 0.0
                            mz = float(sample.z_m) if hasattr(sample, 'z_m') and sample.z_m is not None else 0.0
                            cpx = float(sample.x_a) if hasattr(sample, 'x_a') and sample.x_a is not None else 0.0
                            cpy = float(sample.y_a) if hasattr(sample, 'y_a') and sample.y_a is not None else 0.0
                            cpz = float(sample.z_a) if hasattr(sample, 'z_a') and sample.z_a is not None else 0.0
                            force_data_list.extend([fx, fy, fz, mx, my, mz, cpx, cpy, cpz])
                        else: # Fallback if structure is different or no forces
                            force_data_list.extend([0.0] * 9)
            
            current_row_data.extend(markers_data)
            current_row_data.extend(joint_data)
            current_row_data.extend(force_data_list)

            # Emit header if first packet and headers not yet sent
            if not self._first_packet_processed_for_header:
                header_row = ["Frame", "Timestamp"]
                # 3D Headers
                if qtm_rt.packet.QRTComponentType.Component3d in packet.components:
                    _, markers = packet.get_3d_markers()
                    for i in range(len(markers)):
                        header_row.extend([f"Marker{i+1}_X", f"Marker{i+1}_Y", f"Marker{i+1}_Z"])
                # 6D Headers
                if qtm_rt.packet.QRTComponentType.Component6d in packet.components:
                    _, bodies = packet.get_6d()
                    for i in range(len(bodies)):
                        header_row.extend([f"Body{i+1}_X", f"Body{i+1}_Y", f"Body{i+1}_Z", 
                                           f"Body{i+1}_QW", f"Body{i+1}_QX", f"Body{i+1}_QY", f"Body{i+1}_QZ"])
                # Force Headers
                if qtm_rt.packet.QRTComponentType.ComponentForce in packet.components:
                    force_plates_data_header = packet.get_force()
                    if force_plates_data_header:
                        _, force_plates_header = force_plates_data_header
                        for i in range(len(force_plates_header)):
                            header_row.extend([
                                f"FP{i+1}_FX", f"FP{i+1}_FY", f"FP{i+1}_FZ",
                                f"FP{i+1}_MX", f"FP{i+1}_MY", f"FP{i+1}_MZ",
                                f"FP{i+1}_COPX", f"FP{i+1}_COPY", f"FP{i+1}_COPZ",
                            ])
                self.header_data_ready.emit(header_row)
                self._first_packet_processed_for_header = True
            
            self.data_row_ready.emit(current_row_data)

        except Exception as e:
            tb_str = traceback.format_exc()
            self.error_occurred.emit(f"Error processing QTM packet: {e}\n{tb_str}")

    async def _manage_connection(self):
        self.connection_status.emit(f"Attempting to connect to QTM at {self.host}:{self.port}...")
        try:
            self.qtm_connection = await qtm_rt.connect(self.host, self.port, version="1.22")
            if self.qtm_connection is None:
                self.connection_status.emit("Error: Failed to connect to QTM (connection is None).")
                return

            self.connection_status.emit("Connected to QTM. Checking state...")
            qtm_state = await self.qtm_connection.get_state()
            if qtm_state: self.qtm_event.emit(qtm_state) # Emit if valid state received
            self.connection_status.emit(f"QTM Initial State: {qtm_state.name if hasattr(qtm_state, 'name') else 'Unknown'}")

            if self._running and not self._is_streaming and not self._is_preview_streaming:
                if qtm_state == qtm_rt.QRTEvent.EventCaptureStarted or qtm_state == qtm_rt.QRTEvent.EventCalibrationStarted:
                    self.log_message("QTM already capturing or in calibration. Attempting to start preview stream.")
                    await self._start_preview_stream_async(["3d", "6d"])

            while self._running:
                if not self.qtm_connection or not self.qtm_connection.has_transport():
                    self.connection_status.emit("Error: QTM connection lost.")
                    self._running = False
                    break
                await asyncio.sleep(0.2)
        
        except asyncio.CancelledError:
            self.connection_status.emit("QTM connection task cancelled.")
        # qtm_rt library might raise a general ConnectionError or socket.error for closed connections.
        # Or a specific qtm_rt exception if defined (e.g. qtm_rt.QTMError or similar)
        except (ConnectionError, socket.error) as conn_err: # Catching common connection errors
            self.connection_status.emit(f"QTM Connection Closed/Error: {conn_err}")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"QTM Connection Error: {e}\n{tb_str}"
            self.connection_status.emit(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            if self.qtm_connection and self.qtm_connection.has_transport():
                if self._is_streaming:
                    await self._stop_streaming_async(is_preview=False)
                if self._is_preview_streaming:
                    await self._stop_streaming_async(is_preview=True)
                await self.qtm_connection.disconnect()
                self.connection_status.emit("Disconnected from QTM.")
            self.qtm_connection = None
            self._is_streaming = False
            self._is_preview_streaming = False
            # self._qtm_ready_for_streaming was a MainWindow flag, not directly managed here.

    def run(self):
        self._running = True
        self._first_packet_processed_for_header = False # Reset for new run
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._manage_connection())
        except Exception as e:
             self.error_occurred.emit(f"QTM worker run error: {e}")
        finally:
            if self.loop.is_running():
                self.loop.stop() # Ensure loop is stopped before closing
            self.loop.close()
            self.finished.emit()

    def stop(self):
        self._running = False
        if self.loop and self.loop.is_running():
            if self._is_streaming and self.qtm_connection:
                asyncio.run_coroutine_threadsafe(self._stop_streaming_async(is_preview=False), self.loop)
            if self._is_preview_streaming and self.qtm_connection:
                asyncio.run_coroutine_threadsafe(self._stop_streaming_async(is_preview=True), self.loop)
            if self.qtm_connection and hasattr(self.qtm_connection, 'disconnect'):
                 asyncio.run_coroutine_threadsafe(self.qtm_connection.disconnect(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)

    async def _start_preview_stream_async(self, components):
        if self.qtm_connection and self.qtm_connection.has_transport() and not self._is_streaming and not self._is_preview_streaming:
            try:
                self.connection_status.emit(f"Starting QTM PREVIEW stream with components: {components}")
                self._first_packet_processed_for_header = False # Reset for new stream
                # For preview, we might not want to take control or start new measurement unless necessary.
                # The stream_frames might fail if QTM is not in a proper state.
                await self.qtm_connection.stream_frames(
                    components=components, 
                    on_packet=self._on_packet_internal # Still use the same packet handler
                )
                self._is_preview_streaming = True
                # Emit a specific status for preview?
                self.connection_status.emit("QTM preview streaming started.") 
                return True
            except Exception as e:
                self.error_occurred.emit(f"Error starting QTM preview stream: {e}")
                self._is_preview_streaming = False
        return False

    async def _start_streaming_async(self, components, for_recording=True):
        if self.qtm_connection and self.qtm_connection.has_transport():
            # If a preview stream is running and we want to start a recording stream, stop preview first.
            if self._is_preview_streaming and for_recording:
                self.log_message("Stopping preview stream to start recording stream.")
                await self._stop_streaming_async(is_preview=True)
            
            if not self._is_streaming: # If not already streaming for recording
                try:
                    stream_type = "RECORDING" if for_recording else "GENERAL"
                    self.connection_status.emit(f"Starting QTM {stream_type} stream with components: {components}")
                    self._first_packet_processed_for_header = False 
                    self._current_components = components
                    await self.qtm_connection.stream_frames(
                        components=components, 
                        on_packet=self._on_packet_internal
                    )
                    self._is_streaming = True # Main streaming flag for recording/primary stream
                    self.connection_status.emit(f"QTM {stream_type} streaming started successfully.")
                    return True
                except Exception as e:
                    tb_str = traceback.format_exc()
                    self.error_occurred.emit(f"Error starting QTM stream: {e}\n{tb_str}")
                    self._is_streaming = False
            elif for_recording:
                 self.connection_status.emit("QTM already streaming (for recording). Components might need update.")
                 # Potentially stop and restart if components are different - advanced
        elif not (self.qtm_connection and self.qtm_connection.has_transport()):
            self.error_occurred.emit("Cannot start stream: QTM not connected.")
        return False

    def start_streaming(self, components):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._start_streaming_async(components, for_recording=True), self.loop)
        else:
            self.error_occurred.emit("QTM worker event loop not running. Cannot start stream.")

    async def _stop_streaming_async(self, is_preview=False):
        stream_type_flag = "_is_preview_streaming" if is_preview else "_is_streaming"
        current_status = getattr(self, stream_type_flag, False)
        stream_name = "preview" if is_preview else "main"

        if self.qtm_connection and self.qtm_connection.has_transport() and current_status:
            try:
                # stream_frames_stop should stop any active stream if only one can be active.
                await self.qtm_connection.stream_frames_stop()
                setattr(self, stream_type_flag, False)
                self.connection_status.emit(f"QTM {stream_name} streaming stopped.")
            except Exception as e:
                self.error_occurred.emit(f"Error stopping QTM {stream_name} stream: {e}")
        elif not current_status:
            self.connection_status.emit(f"QTM {stream_name} not currently streaming.")

    def stop_streaming(self):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._stop_streaming_async(is_preview=False), self.loop)
        else:
            self.error_occurred.emit("QTM worker event loop not running. Cannot stop main stream.")

    async def _trigger_and_stream_if_needed_async(self, password, components):
        if not self.qtm_connection or not self.qtm_connection.has_transport():
            self.error_occurred.emit("Cannot trigger/stream: QTM not connected.")
            return

        try:
            if self._is_preview_streaming:
                self.log_message("Stopping preview stream before trigger/new measurement.")
                await self._stop_streaming_async(is_preview=True)

            current_state = await self.qtm_connection.get_state()
            if current_state: self.qtm_event.emit(current_state)
            self.connection_status.emit(f"QTM state before trigger/stream attempt: {current_state.name if hasattr(current_state, 'name') else 'Unknown'}")

            if current_state == qtm_rt.QRTEvent.EventWaitingForTrigger:
                self.connection_status.emit("QTM is waiting for trigger. Attempting to take control and trigger...")
                async with qtm_rt.TakeControl(self.qtm_connection, password):
                    await self.qtm_connection.trig() # This action itself implies a trigger event
                    self.connection_status.emit("QTM triggered.")
                await asyncio.sleep(0.2) # Give QTM time to change state
                current_state = await self.qtm_connection.get_state()
                if current_state: self.qtm_event.emit(current_state)
                self.connection_status.emit(f"QTM state after trigger: {current_state.name if hasattr(current_state, 'name') else 'Unknown'}")
            
            if current_state == qtm_rt.QRTEvent.EventCaptureStarted or current_state == qtm_rt.QRTEvent.EventCalibrationStarted:
                if not self._is_streaming:
                    self.connection_status.emit("QTM capture in progress or ready. Starting RECORDING stream...")
                    await self._start_streaming_async(components, for_recording=True)
                else:
                    self.connection_status.emit("QTM already streaming for recording.")
            elif current_state != qtm_rt.QRTEvent.EventConnected and current_state != qtm_rt.QRTEvent.EventWaitingForTrigger: # Added check for WaitingForTrigger
                # Avoid starting 'new' if it's already waiting for a trigger or already connected and just needs streaming command
                self.connection_status.emit(f"QTM in state {current_state.name if hasattr(current_state, 'name') else 'Unknown'}. Attempting to start new measurement and stream for RECORDING.")
                async with qtm_rt.TakeControl(self.qtm_connection, password):
                    await self.qtm_connection.new() # This implies a 'new measurement' event
                    self.connection_status.emit("New QTM measurement started.")
                await asyncio.sleep(0.1) 
                if not self._is_streaming:
                    await self._start_streaming_async(components, for_recording=True)
            elif not self._is_streaming: # If connected but not in other specific states, try to stream
                self.connection_status.emit(f"QTM state {current_state.name if hasattr(current_state, 'name') else 'Unknown'}. Attempting to start RECORDING stream.")
                await self._start_streaming_async(components, for_recording=True)
            else:
                self.connection_status.emit(f"QTM in state {current_state.name if hasattr(current_state, 'name') else 'Unknown'} or already streaming. Recording stream not (re)started.")

        except Exception as e:
            tb_str = traceback.format_exc()
            self.error_occurred.emit(f"Error in trigger_and_stream_if_needed: {e}\n{tb_str}")

    def extract_and_write_qtm_header(self, packet):
        # Header generation is now part of _on_packet_internal
        # This method could be used to re-emit if needed, or log
        if not self._first_packet_processed_for_header:
             self.log_message("extract_and_write_qtm_header called, but header should be emitted by _on_packet_internal.")
        pass 

    def process_and_record_packet(self, packet, recorder_worker):
        # Data processing and emission of data_row_ready is handled by _on_packet_internal.
        # This method is kept if MainWindow tries to call it, but its role is reduced.
        # The connection to recorder_worker.write_qtm_data is direct via data_row_ready signal.
        pass

    def log_message(self, msg):
        # Utility if direct logging from worker is needed, though signals are preferred for UI.
        print(f"QTM Worker Log: {msg}") 