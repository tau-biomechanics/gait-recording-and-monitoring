from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                             QLineEdit, QPushButton, QGroupBox, QTextEdit, QCheckBox,
                             QFileDialog, QMessageBox, QSpinBox, QFormLayout,
                             QListWidget, QListWidgetItem, QInputDialog, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSlot, QDateTime
from PyQt6.QtGui import QAction, QTextCursor # Corrected import for QAction and Added QTextCursor
import pyqtgraph as pg # Import pyqtgraph

from config_manager import ConfigManager
from qtm_worker import QtmWorker
from insole_worker import InsoleWorker
from recorder_worker import RecorderWorker

MAX_PLOT_POINTS = 500 # Max data points to display on a plot at a time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTM Insole Recorder with Real-time Plots")
        self.setGeometry(50, 50, 1200, 850) # Adjusted size for plots

        self.config_manager = ConfigManager()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_app_layout = QHBoxLayout(self.central_widget)

        self.left_panel_widget = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_panel_widget)
        
        self._create_menu_bar()
        self._create_settings_group() 
        self.recorder_worker = RecorderWorker(self.output_dir_edit.text())
        self._create_status_log_group()
        
        self.left_panel_layout.addWidget(self.settings_group)
        self.left_panel_layout.addWidget(self.status_log_group, 1)

        self._create_plots_area()

        self.main_app_layout.addWidget(self.left_panel_widget, 1) 
        self.main_app_layout.addWidget(self.plots_tab_widget, 2)  

        self.load_settings()
        
        self.qtm_thread = None
        self.qtm_worker = None
        self.insole_thread = None
        self.insole_worker = None

        self._connect_recorder_signals()
        self._is_qtm_connected_and_ready = False
        self._is_insole_listening = False
        self._is_recording = False
        self._first_insole_packet_trigger_pending = False
        
        self._update_ui_states() 

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        save_settings_action = QAction("&Save Settings", self)
        save_settings_action.triggered.connect(self.save_settings)
        file_menu.addAction(save_settings_action)
        load_settings_action = QAction("&Load Settings", self) 
        load_settings_action.triggered.connect(self.load_settings)
        file_menu.addAction(load_settings_action)
        file_menu.addSeparator()
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _create_settings_group(self):
        self.settings_group = QGroupBox("Configuration & Controls") 
        main_settings_layout = QVBoxLayout() 

        # --- QTM Configuration & Controls --- 
        self.qtm_config_group = QGroupBox("QTM Configuration & Controls")
        qtm_form_layout = QFormLayout()
        self.qtm_host_edit = QLineEdit()
        qtm_form_layout.addRow("QTM Host:", self.qtm_host_edit)
        self.qtm_port_edit = QSpinBox()
        self.qtm_port_edit.setRange(1, 65535)
        qtm_form_layout.addRow("QTM Port:", self.qtm_port_edit)
        self.qtm_password_edit = QLineEdit()
        self.qtm_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        qtm_form_layout.addRow("QTM Password:", self.qtm_password_edit)
        qtm_connection_controls_layout = QHBoxLayout()
        self.connect_qtm_button = QPushButton("Connect QTM")
        self.connect_qtm_button.clicked.connect(self._action_connect_qtm)
        qtm_connection_controls_layout.addWidget(self.connect_qtm_button)
        self.disconnect_qtm_button = QPushButton("Disconnect QTM")
        self.disconnect_qtm_button.setEnabled(False) 
        self.disconnect_qtm_button.clicked.connect(self._action_disconnect_qtm)
        qtm_connection_controls_layout.addWidget(self.disconnect_qtm_button)
        qtm_form_layout.addRow("QTM Actions:", qtm_connection_controls_layout)
        self.qtm_config_group.setLayout(qtm_form_layout)
        main_settings_layout.addWidget(self.qtm_config_group)

        # --- Insole Configuration & Controls --- 
        self.insole_config_group = QGroupBox("Insole Configuration & Controls")
        insole_form_layout = QFormLayout()
        self.insole_ip_edit = QLineEdit()
        insole_form_layout.addRow("Insole IP:", self.insole_ip_edit)
        self.insole_port_edit = QSpinBox()
        self.insole_port_edit.setRange(1, 65535)
        insole_form_layout.addRow("Insole Port:", self.insole_port_edit)
        insole_connection_controls_layout = QHBoxLayout()
        self.connect_insole_button = QPushButton("Connect Insole") 
        self.connect_insole_button.clicked.connect(self._action_connect_insole)
        insole_connection_controls_layout.addWidget(self.connect_insole_button)
        self.disconnect_insole_button = QPushButton("Disconnect Insole") 
        self.disconnect_insole_button.setEnabled(False) 
        self.disconnect_insole_button.clicked.connect(self._action_disconnect_insole)
        insole_connection_controls_layout.addWidget(self.disconnect_insole_button)
        insole_form_layout.addRow("Insole Actions:", insole_connection_controls_layout) 
        self.insole_config_group.setLayout(insole_form_layout)
        main_settings_layout.addWidget(self.insole_config_group)

        # --- Recording & Output Section ---
        self.recording_output_group = QGroupBox("Recording & Output")
        recording_output_form_layout = QFormLayout()
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        output_dir_layout.addWidget(self.output_dir_edit)
        self.browse_output_dir_button = QPushButton("Browse...")
        self.browse_output_dir_button.clicked.connect(self._browse_output_dir)
        output_dir_layout.addWidget(self.browse_output_dir_button)
        recording_output_form_layout.addRow("Output Directory:", output_dir_layout)
        recording_actions_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_recording)
        recording_actions_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_recording)
        recording_actions_layout.addWidget(self.stop_button)
        recording_output_form_layout.addRow("Recording Actions:", recording_actions_layout)
        self.recording_output_group.setLayout(recording_output_form_layout)
        main_settings_layout.addWidget(self.recording_output_group)

        # --- Insole Column Headers ---
        self.insole_headers_group = QGroupBox("Insole Column Headers")
        insole_headers_layout = QHBoxLayout()
        self.insole_headers_list_widget = QListWidget()
        self.insole_headers_list_widget.setToolTip("Define the column headers for the insole data CSV file.")
        insole_headers_layout.addWidget(self.insole_headers_list_widget, 3) 
        insole_header_buttons_layout = QVBoxLayout()
        self.add_header_button = QPushButton("Add")
        self.add_header_button.clicked.connect(self._add_insole_header)
        insole_header_buttons_layout.addWidget(self.add_header_button)
        self.edit_header_button = QPushButton("Edit")
        self.edit_header_button.clicked.connect(self._edit_insole_header)
        insole_header_buttons_layout.addWidget(self.edit_header_button)
        self.remove_header_button = QPushButton("Remove")
        self.remove_header_button.clicked.connect(self._remove_insole_header)
        insole_header_buttons_layout.addWidget(self.remove_header_button)
        insole_header_buttons_layout.addStretch(1) 
        self.move_header_up_button = QPushButton("Move Up")
        self.move_header_up_button.clicked.connect(self._move_insole_header_up)
        insole_header_buttons_layout.addWidget(self.move_header_up_button)
        self.move_header_down_button = QPushButton("Move Down")
        self.move_header_down_button.clicked.connect(self._move_insole_header_down)
        insole_header_buttons_layout.addWidget(self.move_header_down_button)
        insole_headers_layout.addLayout(insole_header_buttons_layout, 1) 
        self.insole_headers_group.setLayout(insole_headers_layout)
        main_settings_layout.addWidget(self.insole_headers_group)
        
        # --- Immediate QTM Trigger Checkbox ---
        self.immediate_trigger_checkbox = QCheckBox("Immediate QTM Trigger on Connection")
        self.immediate_trigger_checkbox.setToolTip(
            "If QTM is waiting for a trigger, send it immediately after systems connect, "
            "otherwise trigger on first insole packet (if this box is unchecked)."
        )
        main_settings_layout.addWidget(self.immediate_trigger_checkbox)

        self.settings_group.setLayout(main_settings_layout)

    def _create_status_log_group(self):
        self.status_log_group = QGroupBox("Status & Log")
        status_log_layout = QVBoxLayout()
        status_labels_layout = QHBoxLayout()
        self.qtm_status_label = QLabel("QTM: Disconnected")
        self.qtm_status_label.setStyleSheet("color: red; font-weight: bold;")
        status_labels_layout.addWidget(self.qtm_status_label)
        self.insole_status_label = QLabel("Insole: Not Listening")
        self.insole_status_label.setStyleSheet("color: red; font-weight: bold;")
        status_labels_layout.addWidget(self.insole_status_label)
        self.recording_status_label = QLabel("Recording: Idle")
        self.recording_status_label.setStyleSheet("font-weight: bold;")
        status_labels_layout.addWidget(self.recording_status_label)
        status_log_layout.addLayout(status_labels_layout)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth) 
        status_log_layout.addWidget(self.log_text_edit)
        self.status_log_group.setLayout(status_log_layout)

    def _create_plots_area(self):
        self.plots_tab_widget = QTabWidget()
        self.plots_tab_widget.setToolTip("Real-time data plots from connected systems.")
        self.qtm_kin_plot_widget = pg.PlotWidget(title="QTM Kinematics (Selected Marker)")
        self.qtm_kin_plot_widget.setLabel('left', 'Coordinate Value')
        self.qtm_kin_plot_widget.setLabel('bottom', 'Time (s) or Frame')
        self.qtm_kin_plot_widget.addLegend()
        self.kin_plot_x = self.qtm_kin_plot_widget.plot(pen='r', name='X')
        self.kin_plot_y = self.qtm_kin_plot_widget.plot(pen='g', name='Y')
        self.kin_plot_z = self.qtm_kin_plot_widget.plot(pen='b', name='Z')
        self.plots_tab_widget.addTab(self.qtm_kin_plot_widget, "QTM Kinematics")
        self.plot_kin_data = {'time': [], 'x': [], 'y': [], 'z': []} 
        self.qtm_force_plot_widget = pg.PlotWidget(title="QTM Force (Selected Plate)")
        self.qtm_force_plot_widget.setLabel('left', 'Force (N)')
        self.qtm_force_plot_widget.setLabel('bottom', 'Time (s) or Frame')
        self.qtm_force_plot_widget.addLegend()
        self.force_plot_fx = self.qtm_force_plot_widget.plot(pen='r', name='Fx')
        self.force_plot_fy = self.qtm_force_plot_widget.plot(pen='g', name='Fy')
        self.force_plot_fz = self.qtm_force_plot_widget.plot(pen='b', name='Fz')
        self.plots_tab_widget.addTab(self.qtm_force_plot_widget, "QTM Force")
        self.plot_force_data = {'time': [], 'fx': [], 'fy': [], 'fz': []}
        self.insole_plot_widget = pg.PlotWidget(title="Insole Data (e.g., Total Force)")
        self.insole_plot_widget.setLabel('left', 'Value')
        self.insole_plot_widget.setLabel('bottom', 'Time (s) or Sample')
        self.insole_plot_widget.addLegend()
        self.insole_plot_left_force = self.insole_plot_widget.plot(pen='c', name='Left Total Force')
        self.insole_plot_right_force = self.insole_plot_widget.plot(pen='m', name='Right Total Force')
        self.plots_tab_widget.addTab(self.insole_plot_widget, "Insole Data")
        self.plot_insole_data = {'time': [], 'left_force': [], 'right_force': []}
        self.raw_data_text_edit = QTextEdit()
        self.raw_data_text_edit.setReadOnly(True)
        self.raw_data_text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) 
        self.plots_tab_widget.addTab(self.raw_data_text_edit, "Raw Data Stream")

    def _browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir_edit.text())
        if directory:
            self.output_dir_edit.setText(directory)
            self.recorder_worker.output_dir = directory 
            self.log_message(f"Output directory set to: {directory}")

    def log_message(self, message):
        self.log_text_edit.append(f"[{self._current_time()}] {message}")
        self.log_text_edit.ensureCursorVisible() 

    def _current_time(self):
        return QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz") 

    def load_settings(self):
        self.qtm_host_edit.setText(self.config_manager.get_qtm_host())
        self.qtm_port_edit.setValue(self.config_manager.get_qtm_port())
        self.qtm_password_edit.setText(self.config_manager.get_qtm_password())
        self.insole_ip_edit.setText(self.config_manager.get_insole_ip())
        self.insole_port_edit.setValue(self.config_manager.get_insole_port())
        self.output_dir_edit.setText(self.config_manager.get_output_dir())
        self.immediate_trigger_checkbox.setChecked(self.config_manager.get_immediate_trigger())
        self.insole_headers_list_widget.clear()
        headers = self.config_manager.get_insole_headers()
        for header in headers:
            self.insole_headers_list_widget.addItem(QListWidgetItem(header))
        if self.recorder_worker:
            self.recorder_worker.output_dir = self.output_dir_edit.text()
        self.log_message("Settings loaded.")

    def save_settings(self):
        self.config_manager.set_qtm_host(self.qtm_host_edit.text())
        self.config_manager.set_qtm_port(self.qtm_port_edit.value())
        self.config_manager.set_qtm_password(self.qtm_password_edit.text())
        self.config_manager.set_insole_ip(self.insole_ip_edit.text())
        self.config_manager.set_insole_port(self.insole_port_edit.value())
        self.config_manager.set_output_dir(self.output_dir_edit.text())
        self.config_manager.set_immediate_trigger(self.immediate_trigger_checkbox.isChecked())
        headers_list = []
        for i in range(self.insole_headers_list_widget.count()):
            item = self.insole_headers_list_widget.item(i)
            if item and item.text().strip(): 
                headers_list.append(item.text().strip())
        if headers_list:
             self.config_manager.set_insole_headers(headers_list)
        else: 
            default_headers_list = self.config_manager.get_insole_headers() 
            self.config_manager.set_insole_headers(default_headers_list) 
            self.insole_headers_list_widget.clear()
            for header in default_headers_list:
                self.insole_headers_list_widget.addItem(QListWidgetItem(header))
        self.config_manager.save_settings()
        self.log_message("Settings saved.")

    def _show_about_dialog(self):
        QMessageBox.about(self, "About QTM Insole Recorder",
                          "QTM Insole Recorder v0.1.0\\n\\n"
                          "This application streams and records data from Qualisys Track Manager (QTM) "
                          "and Moticon Insoles.\\n\\n"
                          "Please ensure PyQt6, qtm-rt are installed.")

    def closeEvent(self, event):
        self.log_message("Closing application...")
        self._action_disconnect_qtm()   # Disconnect QTM individually
        self._action_disconnect_insole() # Disconnect Insole individually
        self.save_settings()
        super().closeEvent(event)

    def _connect_recorder_signals(self):
        self.recorder_worker.file_paths_created.connect(
            lambda qtm_f, insole_f: self.log_message(f"Recording files created: QTM: {qtm_f}, Insole: {insole_f}")
        )
        self.recorder_worker.status_update.connect(self.log_message)
        self.recorder_worker.error_occurred.connect(lambda err: self.log_message(f"RECORDER ERROR: {err}"))

    def _action_connect_qtm(self):
        self.log_message("Attempting to connect QTM system...")
        if self.qtm_worker:
            self.log_message("QTM worker already exists or connection in progress. Skipping.")
            return
        self.save_settings() # Save settings before individual connection attempt

        self.qtm_thread = QThread(self)
        self.qtm_worker = QtmWorker(self.qtm_host_edit.text(), self.qtm_port_edit.value(), self.qtm_password_edit.text())
        self.qtm_worker.moveToThread(self.qtm_thread)
        self.qtm_worker.connection_status.connect(self._on_qtm_connection_status)
        self.qtm_worker.qtm_data_packet.connect(self._on_qtm_data_packet)
        self.qtm_worker.qtm_event.connect(self._on_qtm_event)
        self.qtm_worker.error_occurred.connect(self._on_qtm_error)
        if self.recorder_worker: 
            self.qtm_worker.header_data_ready.connect(self.recorder_worker.write_qtm_header)
            self.qtm_worker.data_row_ready.connect(self.recorder_worker.write_qtm_data)
        self.qtm_thread.started.connect(self.qtm_worker.run)
        self.qtm_worker.finished.connect(self._on_qtm_worker_finished)
        self.qtm_thread.start()
        self.log_message("QTM connection process initiated.")
        self._update_ui_states() 

    def _action_connect_insole(self):
        self.log_message("Attempting to connect Insole system...")
        if self.insole_worker:
            self.log_message("Insole worker already exists or connection in progress. Skipping.")
            return
        self.save_settings() # Save settings before individual connection attempt

        self.insole_thread = QThread(self)
        self.insole_worker = InsoleWorker(self.insole_ip_edit.text(), self.insole_port_edit.value())
        self.insole_worker.moveToThread(self.insole_thread)
        self.insole_worker.connection_status.connect(self._on_insole_connection_status)
        self.insole_worker.insole_data_packet.connect(self._on_insole_data_packet) 
        self.insole_worker.insole_data_packet.connect(self._update_insole_plot)    
        self.insole_worker.first_packet_received.connect(self._on_first_insole_packet)
        self.insole_worker.error_occurred.connect(self._on_insole_error)
        self.insole_thread.started.connect(self.insole_worker.run)
        self.insole_worker.finished.connect(self._on_insole_worker_finished)
        self.insole_thread.start()
        self.log_message("Insole connection process initiated.")
        self._update_ui_states() 

    def _action_disconnect_qtm(self):
        self.log_message("Disconnecting QTM system...")
        if self._is_recording and self.qtm_worker:
            QMessageBox.warning(self, "Recording Active", "Cannot disconnect QTM while recording. Please stop recording first.")
            return

        if self.qtm_worker:
            self.qtm_worker.stop() 
            self.log_message("QTM stop signal sent.")
        else:
            self.log_message("QTM system not connected or already disconnecting.")
        self._update_ui_states() 

    def _action_disconnect_insole(self):
        self.log_message("Disconnecting Insole system...")
        if self._is_recording and self.insole_worker:
            QMessageBox.warning(self, "Recording Active", "Cannot disconnect Insole while recording. Please stop recording first.")
            return

        if self.insole_worker:
            self.insole_worker.stop() 
            self.log_message("Insole stop signal sent.")
        else:
            self.log_message("Insole system not connected or already disconnecting.")
        self._update_ui_states() 
            
    def _start_recording(self):
        if not (self._is_qtm_connected_and_ready and self._is_insole_listening):
            QMessageBox.warning(self, "Not Ready", "Both QTM and Insole systems must be connected and ready for recording.")
            return
        if self._is_recording:
            self.log_message("Already recording.")
            return

        self.log_message("Starting recording...")
        self.save_settings() # Ensure latest settings are used for recording
        if self.recorder_worker:
            self.recorder_worker.output_dir = self.output_dir_edit.text()
            current_insole_headers = []
            for i in range(self.insole_headers_list_widget.count()):
                item = self.insole_headers_list_widget.item(i)
                if item and item.text().strip():
                    current_insole_headers.append(item.text().strip())
            if not current_insole_headers: 
                self.log_message("Insole headers list is empty, using default headers from configuration for recording.")
                current_insole_headers = self.config_manager.get_insole_headers()
                self.insole_headers_list_widget.clear()
                for header in current_insole_headers:
                    self.insole_headers_list_widget.addItem(QListWidgetItem(header))
            self.recorder_worker.start_recording(insole_headers_list=current_insole_headers)
        
        if self.qtm_worker:
            qtm_components_to_stream = self._get_qtm_components()
            self.log_message(f"Requesting QTM to start streaming components: {qtm_components_to_stream}")
            self.qtm_worker.start_streaming(qtm_components_to_stream)

        self._is_recording = True
        self._first_insole_packet_trigger_pending = False 
        self._update_ui_states()
        self.log_message("Recording initiated. Waiting for data...")

    def _stop_recording(self):
        self.log_message("Stopping recording...")
        if self.qtm_worker:
             self.qtm_worker.stop_streaming() 
        self.recorder_worker.stop_recording()
        self._is_recording = False
        self._first_insole_packet_trigger_pending = False 
        self._update_ui_states() 
        self.log_message("Recording stopped.")

    @pyqtSlot(str)
    def _on_qtm_connection_status(self, status):
        self.log_message(f"QTM Status: {status}")
        if "Connected to QTM" in status or "QTM state after trigger" in status or "streaming started successfully" in status or "preview streaming started" in status:
            self._is_qtm_connected_and_ready = True 
            if self.immediate_trigger_checkbox.isChecked() and self.qtm_worker:
                self.log_message("Immediate trigger is checked. QTM worker should handle triggering if needed.")
        elif "Error" in status or "Disconnected" in status or "connection lost" in status or "connection task cancelled" in status or "Connection Closed" in status:
            self._is_qtm_connected_and_ready = False
            if self._is_recording:
                self._stop_recording()
                QMessageBox.critical(self, "QTM Error", f"QTM connection lost: {status}. Recording stopped.")
        self._update_ui_states() 

    @pyqtSlot(object)
    def _on_qtm_data_packet(self, packet): 
        if hasattr(self, 'raw_data_text_edit'):
            try:
                log_msg = f"[{self._current_time()}] QTM Packet: Frame {packet.framenumber}, TS {packet.timestamp}"
                if hasattr(packet, 'components') and packet.components:
                    component_names = [comp.name for comp in packet.components.keys()]
                    log_msg += f", Components: {component_names}"
                self.raw_data_text_edit.append(log_msg)
                if self.raw_data_text_edit.document().lineCount() > 2000: 
                    cursor = self.raw_data_text_edit.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.Start)
                    cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                    cursor.removeSelectedText()
                    cursor.deletePreviousChar() 
                    self.raw_data_text_edit.setTextCursor(cursor)
            except Exception as log_e:
                self.raw_data_text_edit.append(f"[{self._current_time()}] QTM Packet: Error logging packet details: {log_e}")
        
        if self._is_recording and self.recorder_worker:
            if not self.recorder_worker._qtm_headers_written:
                if self.qtm_worker: 
                    self.qtm_worker.extract_and_write_qtm_header(packet) 
            
            if self.qtm_worker: 
                self.qtm_worker.process_and_record_packet(packet, self.recorder_worker)

        self._update_qtm_plots(packet)

    @pyqtSlot(object) 
    def _on_qtm_event(self, event_info):
        self.log_message(f"QTM Event: {event_info}")
        self._update_ui_states() 

    @pyqtSlot(str)
    def _on_qtm_error(self, error_message):
        self.log_message(f"QTM WORKER ERROR: {error_message}")
        self._is_qtm_connected_and_ready = False 
        self._update_ui_states()

    @pyqtSlot(str)
    def _on_insole_connection_status(self, status):
        self.log_message(f"Insole Status: {status}")
        if "Listening" in status:
            self._is_insole_listening = True
        elif "Error" in status or "stopped" in status:
            self._is_insole_listening = False
            if self._is_recording:
                self.log_message("Insole listener stopped or errored during recording.")
        self._update_ui_states()

    @pyqtSlot(list)
    def _on_insole_data_packet(self, data_list):
        if hasattr(self, 'raw_data_text_edit'):
            self.raw_data_text_edit.append(f"[{self._current_time()}] INSOLE: {data_list}")
            if self.raw_data_text_edit.document().lineCount() > 2000: 
                cursor = self.raw_data_text_edit.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deletePreviousChar() 
                self.raw_data_text_edit.setTextCursor(cursor)
        if self._is_recording and self.recorder_worker:
            if not self.recorder_worker._insole_headers_written:
                self.log_message("Insole header not written by start_recording. Data might be lost or misaligned.")
            self.recorder_worker.write_insole_data(data_list)

    @pyqtSlot()
    def _on_first_insole_packet(self):
        self.log_message("First insole packet received.")
        if not self.immediate_trigger_checkbox.isChecked() and self.qtm_worker and self._is_qtm_connected_and_ready:
            if not self._is_recording: 
                 self._first_insole_packet_trigger_pending = True
                 self.log_message("Trigger pending for QTM on first insole packet. Start recording to activate.")
            else: 
                self.log_message("Attempting to trigger QTM or start streaming due to first insole packet while recording.")
                if self.qtm_worker:
                    self.qtm_worker.trigger_and_stream_if_needed(self.qtm_password_edit.text(), self._get_qtm_components())
        self._update_ui_states() 

    @pyqtSlot(str)
    def _on_insole_error(self, error_message):
        self.log_message(f"INSOLE WORKER ERROR: {error_message}")
        self._is_insole_listening = False 
        self._update_ui_states()

    @pyqtSlot()
    def _on_qtm_worker_finished(self):
        self.log_message("QTM worker thread finished.")
        if self.qtm_thread:
            self.qtm_thread.quit()
            self.qtm_thread.wait() 
        self.qtm_thread = None 
        self.qtm_worker = None 
        self._is_qtm_connected_and_ready = False 
        self._update_ui_states() 

    @pyqtSlot()
    def _on_insole_worker_finished(self):
        self.log_message("Insole worker thread finished.")
        if self.insole_thread:
            self.insole_thread.quit()
            self.insole_thread.wait() 
        self.insole_thread = None 
        self.insole_worker = None 
        self._is_insole_listening = False 
        self._update_ui_states() 

    def _set_settings_enabled(self, enabled):
        # This method enables/disables individual form elements.
        # The group boxes themselves are handled in _update_ui_states.
        self.qtm_host_edit.setEnabled(enabled)
        self.qtm_port_edit.setEnabled(enabled)
        self.qtm_password_edit.setEnabled(enabled)
        self.insole_ip_edit.setEnabled(enabled)
        self.insole_port_edit.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled) 
        self.browse_output_dir_button.setEnabled(enabled)
        self.insole_headers_list_widget.setEnabled(enabled)
        self.immediate_trigger_checkbox.setEnabled(enabled)
        self.add_header_button.setEnabled(enabled)
        self.edit_header_button.setEnabled(enabled)
        self.remove_header_button.setEnabled(enabled)
        self.move_header_up_button.setEnabled(enabled)
        self.move_header_down_button.setEnabled(enabled)

    def _get_qtm_components(self):
        return ["3d", "6d", "force"] 

    def _update_ui_states(self):
        qtm_worker_exists = self.qtm_worker is not None
        insole_worker_exists = self.insole_worker is not None

        # --- Group Box Enabled/Disabled Logic ---
        # Settings are editable if NO worker is active AND not recording.
        settings_overall_editable = not qtm_worker_exists and not insole_worker_exists and not self._is_recording
        
        # Enable/disable entire group boxes first
        # self.settings_group.setEnabled(settings_overall_editable) # This is the main container, too broad.

        # Enable/disable QTM config group
        # QTM settings fields are enabled if QTM is not active and not recording.
        # QTM connect button is enabled if QTM is not active and not recording.
        # QTM disconnect button is enabled if QTM is active and not recording.
        qtm_config_group_editable = not qtm_worker_exists and not self._is_recording
        self.qtm_config_group.setEnabled(qtm_config_group_editable)
        self.connect_qtm_button.setEnabled(not qtm_worker_exists and not self._is_recording) 
        self.disconnect_qtm_button.setEnabled(qtm_worker_exists and not self._is_recording)

        # Enable/disable Insole config group
        insole_config_group_editable = not insole_worker_exists and not self._is_recording
        self.insole_config_group.setEnabled(insole_config_group_editable)
        self.connect_insole_button.setEnabled(not insole_worker_exists and not self._is_recording)
        self.disconnect_insole_button.setEnabled(insole_worker_exists and not self._is_recording)

        # Enable/disable Recording & Output group (Output dir and recording buttons)
        # Recording buttons depend on QTM & Insole readiness.
        # Output dir browse button should be editable if not recording and no workers active.
        recording_output_group_editable = settings_overall_editable # For output directory part
        self.recording_output_group.setEnabled(True) # Keep the group box itself always enabled
        self.output_dir_edit.setEnabled(recording_output_group_editable)
        self.browse_output_dir_button.setEnabled(recording_output_group_editable)

        # Other settings like Insole Headers, Immediate Trigger
        self.insole_headers_group.setEnabled(settings_overall_editable)
        self.immediate_trigger_checkbox.setEnabled(settings_overall_editable)

        # --- Individual Button Logic (within already handled groups) ---
        # Connect/Disconnect All buttons are removed.
        
        # Start Recording: if QTM is ready AND Insole is listening AND not already recording
        can_start_recording = self._is_qtm_connected_and_ready and self._is_insole_listening and not self._is_recording
        self.start_button.setEnabled(can_start_recording)
        # Stop Recording: if currently recording
        self.stop_button.setEnabled(self._is_recording)
        
        # --- Status Label Logic ---
        if self._is_qtm_connected_and_ready:
            qtm_status_text = "QTM: Connected & Ready"
            qtm_status_style = "color: green; font-weight: bold;"
            if self.qtm_worker and hasattr(self.qtm_worker, '_is_streaming') and self.qtm_worker._is_streaming:
                qtm_status_text = "QTM: Streaming"
                qtm_status_style = "color: darkGreen; font-weight: bold;"
            elif hasattr(self.qtm_worker, '_is_preview_streaming') and self.qtm_worker._is_preview_streaming:
                qtm_status_text = "QTM: Previewing"
                qtm_status_style = "color: teal; font-weight: bold;"
        elif qtm_worker_exists: 
             qtm_status_text = "QTM: Connecting..."
             qtm_status_style = "color: orange; font-weight: bold;"
        else: 
            qtm_status_text = "QTM: Disconnected"
            qtm_status_style = "color: red; font-weight: bold;"
        self.qtm_status_label.setText(qtm_status_text)
        self.qtm_status_label.setStyleSheet(qtm_status_style)

        if self._is_insole_listening:
            insole_status_text = "Insole: Listening"
            insole_status_style = "color: green; font-weight: bold;"
        elif insole_worker_exists: 
            insole_status_text = "Insole: Initializing..."
            insole_status_style = "color: orange; font-weight: bold;"
        else: 
            insole_status_text = "Insole: Not Listening"
            insole_status_style = "color: red; font-weight: bold;"
        self.insole_status_label.setText(insole_status_text)
        self.insole_status_label.setStyleSheet(insole_status_style)

        if self._is_recording:
            self.recording_status_label.setText("Recording: Active")
            self.recording_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.recording_status_label.setText("Recording: Idle")
            self.recording_status_label.setStyleSheet("font-weight: bold;")

    def _add_insole_header(self):
        text, ok = QInputDialog.getText(self, "Add Insole Header", "Header name:")
        if ok and text.strip():
            self.insole_headers_list_widget.addItem(QListWidgetItem(text.strip()))
            self.log_message(f"Insole header added: {text.strip()}")
        elif ok and not text.strip():
            QMessageBox.warning(self, "Empty Header", "Header name cannot be empty.")

    def _edit_insole_header(self):
        current_item = self.insole_headers_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a header to edit.")
            return
        old_text = current_item.text()
        text, ok = QInputDialog.getText(self, "Edit Insole Header", "New header name:", QLineEdit.EchoMode.Normal, old_text)
        if ok and text.strip():
            current_item.setText(text.strip())
            self.log_message(f"Insole header edited from '{old_text}' to '{text.strip()}'")
        elif ok and not text.strip():
            QMessageBox.warning(self, "Empty Header", "Header name cannot be empty.")

    def _remove_insole_header(self):
        current_item = self.insole_headers_list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a header to remove.")
            return
        row = self.insole_headers_list_widget.row(current_item)
        item = self.insole_headers_list_widget.takeItem(row)
        self.log_message(f"Insole header removed: {item.text()}")
        del item

    def _move_insole_header_up(self):
        current_row = self.insole_headers_list_widget.currentRow()
        if current_row > 0: 
            item = self.insole_headers_list_widget.takeItem(current_row)
            self.insole_headers_list_widget.insertItem(current_row - 1, item)
            self.insole_headers_list_widget.setCurrentRow(current_row - 1)
            self.log_message(f"Moved header '{item.text()}' up.")

    def _move_insole_header_down(self):
        current_row = self.insole_headers_list_widget.currentRow()
        if current_row < self.insole_headers_list_widget.count() - 1: 
            item = self.insole_headers_list_widget.takeItem(current_row)
            self.insole_headers_list_widget.insertItem(current_row + 1, item)
            self.insole_headers_list_widget.setCurrentRow(current_row + 1)
            self.log_message(f"Moved header '{item.text()}' down.")

    @pyqtSlot(object) 
    def _update_qtm_plots(self, qtm_packet):
        try:
            if hasattr(qtm_packet, "get_3d_markers") and qtm_packet.get_3d_markers():
                _, markers = qtm_packet.get_3d_markers()
                if markers:
                    marker0 = markers[0]
                    current_time = qtm_packet.timestamp / 1_000_000.0 
                    self.plot_kin_data['time'].append(current_time)
                    self.plot_kin_data['x'].append(float(marker0.x) if marker0.x is not None else 0)
                    self.plot_kin_data['y'].append(float(marker0.y) if marker0.y is not None else 0)
                    self.plot_kin_data['z'].append(float(marker0.z) if marker0.z is not None else 0)
                    for key in self.plot_kin_data:
                        self.plot_kin_data[key] = self.plot_kin_data[key][-MAX_PLOT_POINTS:]
                    self.kin_plot_x.setData(self.plot_kin_data['time'], self.plot_kin_data['x'])
                    self.kin_plot_y.setData(self.plot_kin_data['time'], self.plot_kin_data['y'])
                    self.kin_plot_z.setData(self.plot_kin_data['time'], self.plot_kin_data['z'])
            if hasattr(qtm_packet, "get_force") and qtm_packet.get_force():
                _, force_plates = qtm_packet.get_force()
                if force_plates and hasattr(force_plates[0], 'forces') and force_plates[0].forces:
                    force_sample = force_plates[0].forces[0]
                    current_time = qtm_packet.timestamp / 1_000_000.0
                    self.plot_force_data['time'].append(current_time)
                    self.plot_force_data['fx'].append(float(force_sample.x) if hasattr(force_sample, 'x') and force_sample.x is not None else 0)
                    self.plot_force_data['fy'].append(float(force_sample.y) if hasattr(force_sample, 'y') and force_sample.y is not None else 0)
                    self.plot_force_data['fz'].append(float(force_sample.z) if hasattr(force_sample, 'z') and force_sample.z is not None else 0)
                    for key in self.plot_force_data:
                        self.plot_force_data[key] = self.plot_force_data[key][-MAX_PLOT_POINTS:]
                    self.force_plot_fx.setData(self.plot_force_data['time'], self.plot_force_data['fx'])
                    self.force_plot_fy.setData(self.plot_force_data['time'], self.plot_force_data['fy'])
                    self.force_plot_fz.setData(self.plot_force_data['time'], self.plot_force_data['fz'])
        except Exception as e:
            self.log_message(f"Error updating QTM plot: {e}")

    @pyqtSlot(list) 
    def _update_insole_plot(self, insole_data):
        try:
            if len(insole_data) >= 3: 
                current_time = float(insole_data[0]) 
                l_force = float(insole_data[1])
                r_force = float(insole_data[2])
                self.plot_insole_data['time'].append(current_time)
                self.plot_insole_data['left_force'].append(l_force)
                self.plot_insole_data['right_force'].append(r_force)
                for key in self.plot_insole_data:
                    self.plot_insole_data[key] = self.plot_insole_data[key][-MAX_PLOT_POINTS:]
                self.insole_plot_left_force.setData(self.plot_insole_data['time'], self.plot_insole_data['left_force'])
                self.insole_plot_right_force.setData(self.plot_insole_data['time'], self.plot_insole_data['right_force'])
        except Exception as e:
            self.log_message(f"Error updating Insole plot: {e}")

if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication # Added import
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 