from PyQt6.QtCore import QSettings

APP_NAME = "QTMInsoleRecorder"
ORGANIZATION_NAME = "MyOrganization" # Replace if desired

# Configuration keys
KEY_QTM_HOST = "qtm/host"
KEY_QTM_PORT = "qtm/port"
KEY_QTM_PASSWORD = "qtm/password"
KEY_INSOLE_IP = "insole/ip"
KEY_INSOLE_PORT = "insole/port"
KEY_OUTPUT_DIR = "output/directory"
KEY_IMMEDIATE_TRIGGER = "settings/immediate_trigger"
KEY_INSOLE_HEADERS = "insole/headers" # Stored as a comma-separated string

class ConfigManager:
    def __init__(self):
        self.settings = QSettings(ORGANIZATION_NAME, APP_NAME)

    def get_qtm_host(self, default="127.0.0.1"):
        return self.settings.value(KEY_QTM_HOST, default)

    def set_qtm_host(self, host):
        self.settings.setValue(KEY_QTM_HOST, host)

    def get_qtm_port(self, default=22223):
        return int(self.settings.value(KEY_QTM_PORT, default))

    def set_qtm_port(self, port):
        self.settings.setValue(KEY_QTM_PORT, port)

    def get_qtm_password(self, default=""):
        return self.settings.value(KEY_QTM_PASSWORD, default)

    def set_qtm_password(self, password):
        self.settings.setValue(KEY_QTM_PASSWORD, password)

    def get_insole_ip(self, default="0.0.0.0"):
        return self.settings.value(KEY_INSOLE_IP, default)

    def set_insole_ip(self, ip):
        self.settings.setValue(KEY_INSOLE_IP, ip)

    def get_insole_port(self, default=5555):
        return int(self.settings.value(KEY_INSOLE_PORT, default))

    def set_insole_port(self, port):
        self.settings.setValue(KEY_INSOLE_PORT, port)

    def get_output_dir(self, default="data"):
        return self.settings.value(KEY_OUTPUT_DIR, default)

    def set_output_dir(self, directory):
        self.settings.setValue(KEY_OUTPUT_DIR, directory)

    def get_immediate_trigger(self, default=False):
        return self.settings.value(KEY_IMMEDIATE_TRIGGER, default, type=bool)

    def set_immediate_trigger(self, enabled):
        self.settings.setValue(KEY_IMMEDIATE_TRIGGER, enabled)
        
    def get_insole_headers(self, default_list=None):
        if default_list is None:
            default_list = [
                "Timestamp", "Left_Total_Force", "Right_Total_Force",
                "Left_COP_X", "Left_COP_Y", "Right_COP_X", "Right_COP_Y",
                "Stance_Phase", "Gait_Line"
            ]
        headers_str = self.settings.value(KEY_INSOLE_HEADERS, ",".join(default_list))
        return headers_str.split(',') if headers_str else default_list

    def set_insole_headers(self, headers_list):
        self.settings.setValue(KEY_INSOLE_HEADERS, ",".join(headers_list))

    def save_settings(self):
        self.settings.sync() 