from PyQt6.QtCore import QObject, pyqtSignal
import csv
import os
import datetime

class RecorderWorker(QObject):
    file_paths_created = pyqtSignal(str, str) # qtm_file_path, insole_file_path
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self._running = False
        self.qtm_output_file = None
        self.insole_output_file = None
        self.qtm_csv_writer = None
        self.insole_csv_writer = None
        self.qtm_file_handle = None
        self.insole_file_handle = None
        self._insole_headers_written = False
        self._qtm_headers_written = False

    def start_recording(self, insole_headers_list=None):
        self._running = True
        self.status_update.emit("Recorder worker started.")
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.status_update.emit(f"Created output directory: {self.output_dir}")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.qtm_output_file = os.path.join(self.output_dir, f"qtm_data_{timestamp}.csv")
            self.insole_output_file = os.path.join(self.output_dir, f"insole_data_{timestamp}.csv")

            self.qtm_file_handle = open(self.qtm_output_file, "w", newline="")
            self.qtm_csv_writer = csv.writer(self.qtm_file_handle)

            self.insole_file_handle = open(self.insole_output_file, "w", newline="")
            self.insole_csv_writer = csv.writer(self.insole_file_handle)
            
            self.file_paths_created.emit(self.qtm_output_file, self.insole_output_file)
            self.status_update.emit(f"Saving QTM data to: {self.qtm_output_file}")
            self.status_update.emit(f"Saving Insole data to: {self.insole_output_file}")

            if insole_headers_list:
                self.write_insole_header(insole_headers_list)
            
            self._insole_headers_written = False # Reset in case of multiple recordings in one session
            self._qtm_headers_written = False

        except Exception as e:
            err_msg = f"Error initializing recorder: {e}"
            self.error_occurred.emit(err_msg)
            self.status_update.emit(err_msg)
            self._running = False # Don't proceed if setup failed

    def write_qtm_header(self, header_row):
        if not self._running or not self.qtm_csv_writer:
            self.error_occurred.emit("QTM recorder not ready or stopped.")
            return
        if not self._qtm_headers_written:
            try:
                self.qtm_csv_writer.writerow(header_row)
                self.qtm_file_handle.flush() # Ensure header is written immediately
                self._qtm_headers_written = True
                self.status_update.emit("QTM data header written.")
            except Exception as e:
                self.error_occurred.emit(f"Error writing QTM header: {e}")
        
    def write_qtm_data(self, data_row):
        if not self._running or not self.qtm_csv_writer:
            # self.error_occurred.emit("QTM recorder not ready or stopped for data writing.")
            return
        if not self._qtm_headers_written:
            # self.error_occurred.emit("QTM header not written. Cannot write data.")
            # This could be noisy; maybe log once or handle by ensuring header is always written first.
            return
        try:
            self.qtm_csv_writer.writerow(data_row)
        except Exception as e:
            self.error_occurred.emit(f"Error writing QTM data: {e}")

    def write_insole_header(self, header_row):
        if not self._running or not self.insole_csv_writer:
            self.error_occurred.emit("Insole recorder not ready or stopped.")
            return
        if not self._insole_headers_written:
            try:
                self.insole_csv_writer.writerow(header_row)
                self.insole_file_handle.flush()
                self._insole_headers_written = True
                self.status_update.emit("Insole data header written.")
            except Exception as e:
                self.error_occurred.emit(f"Error writing insole header: {e}")

    def write_insole_data(self, data_row):
        if not self._running or not self.insole_csv_writer:
            # self.error_occurred.emit("Insole recorder not ready or stopped for data writing.")
            return
        if not self._insole_headers_written:
            # self.error_occurred.emit("Insole header not written. Defaulting or erroring may be needed.")
            # Fallback to default if no custom headers were explicitly set via start_recording
            default_headers = [
                "Timestamp", "Left_Total_Force", "Right_Total_Force", 
                "Left_COP_X", "Left_COP_Y", "Right_COP_X", "Right_COP_Y",
                "Stance_Phase", "Gait_Line" # Example, match qtm_stream.py
            ]
            # Ensure this logic aligns with how headers are managed overall.
            # self.write_insole_header(default_headers) # Be careful about re-writing.
            self.error_occurred.emit("Attempted to write insole data before header.")
            return

        try:
            self.insole_csv_writer.writerow(data_row)
        except Exception as e:
            self.error_occurred.emit(f"Error writing insole data: {e}")

    def stop_recording(self):
        self._running = False
        self.status_update.emit("Stopping recorder worker...")
        try:
            if self.qtm_file_handle and not self.qtm_file_handle.closed:
                self.qtm_file_handle.flush() # Ensure all data is written
                self.qtm_file_handle.close()
                self.status_update.emit(f"QTM data saved to {self.qtm_output_file}")
            if self.insole_file_handle and not self.insole_file_handle.closed:
                self.insole_file_handle.flush()
                self.insole_file_handle.close()
                self.status_update.emit(f"Insole data saved to {self.insole_output_file}")
        except Exception as e:
            self.error_occurred.emit(f"Error closing files: {e}")
        finally:
            self.qtm_file_handle = None
            self.insole_file_handle = None
            self.qtm_csv_writer = None
            self.insole_csv_writer = None
            self._insole_headers_written = False
            self._qtm_headers_written = False
            self.status_update.emit("Recorder worker finished.")
            self.finished.emit()

    # This worker is designed to be called directly from the main thread or other workers' signals
    # rather than running its own loop in a QThread for now.
    # If file I/O becomes blocking, it could be moved to a QThread.
    # For now, its methods are slots to be connected. 