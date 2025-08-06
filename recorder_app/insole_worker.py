from PyQt6.QtCore import QObject, pyqtSignal, QThread
import socket
import time # For sleep

class InsoleWorker(QObject):
    insole_data_packet = pyqtSignal(list) # Emits parsed insole data (e.g., list of floats)
    connection_status = pyqtSignal(str) # e.g., "Listening", "Stopped", "Error: ..."
    error_occurred = pyqtSignal(str)
    first_packet_received = pyqtSignal() # To signal QTM trigger if needed
    finished = pyqtSignal()

    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self._running = False
        self.sock = None

    def run(self):
        self._running = True
        self.connection_status.emit(f"Starting UDP listener on {self.ip}:{self.port}")
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Use "0.0.0.0" to listen on all interfaces if ip is 127.0.0.1 or 0.0.0.0
            listen_ip = "0.0.0.0" if self.ip == "127.0.0.1" or self.ip == "0.0.0.0" else self.ip
            self.sock.bind((listen_ip, self.port))
            self.sock.settimeout(0.5) # Timeout for recvfrom to allow checking self._running
            self.connection_status.emit(f"Successfully bound to {listen_ip}:{self.port}. Listening...")
            
            _first_packet = True
            while self._running:
                try:
                    data, addr = self.sock.recvfrom(2048) # Buffer size
                    # self.connection_status.emit(f"Received UDP packet from {addr[0]}:{addr[1]}")
                    
                    data_str = data.decode("utf-8").strip()
                    values = data_str.split()
                    try:
                        parsed_values = [float(v) for v in values]
                        self.insole_data_packet.emit(parsed_values)
                        if _first_packet:
                            self.first_packet_received.emit()
                            _first_packet = False
                    except ValueError:
                        self.error_occurred.emit(f"Could not parse insole data: {data_str}")

                except socket.timeout:
                    continue # Just to check self._running flag
                except Exception as e:
                    self.error_occurred.emit(f"Insole listener error: {e}")
                    # Short pause before trying again or consider stopping
                    time.sleep(0.1)
            
        except Exception as e:
            error_msg = f"Error setting up insole listener: {e}"
            self.connection_status.emit(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            if self.sock:
                self.sock.close()
            self.connection_status.emit("Insole listener stopped.")
            self.finished.emit()

    def stop(self):
        self._running = False
        # self.finished.emit() # Emitted in run() finally block 