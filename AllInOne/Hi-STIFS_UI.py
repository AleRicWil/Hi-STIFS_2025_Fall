import tkinter as tk
from tkinter import ttk, filedialog
import serial.tools.list_ports
from collect_data import run_collection
from calibration import run_calibration, calculate_coefficients, run_accel_calibration
from process import process_data, show_force_position
from multiprocessing import Process, Queue
import time
from datetime import datetime
import csv
import os
import json

class HardwareControlUI:
    """Main UI class for controlling Hi-STIFS hardware and collecting strain data.

    Attributes:
        root (tk.Tk): The main Tkinter window.
        container (ttk.Frame): Frame to hold different pages.
        pages (dict): Dictionary mapping page names to their frames.
        status_queue (Queue): Queue for receiving status messages from collection process.
        collection_process (Process): Current data collection process, if running.
        last_config (dict): Last used configuration for header fields.
        header_fields (list): List of tuples defining header fields, their keys, and types.
    """

    header_fields = [
        ("Hi-STIFS configuration", "configuration", str),
        ("stalk array (lo,med,hi)", "pvc_stiffness", str),
        ("sensor height (cm)", "height", float),
        ("sensor yaw (degrees)", "yaw", float),
        ("sensor pitch (degrees)", "pitch", float),
        ("sensor roll (degrees)", "roll", float),
        ("rate of travel (ft/min)", "rate_of_travel", float),
        ("angle of travel (degrees)", "angle_of_travel", float),
        ("offset distance (cm to gauge 2)", "offset_distance", float),
    ]

    def __init__(self, root):
        """Initialize the UI.

        Args:
            root (tk.Tk): The Tkinter root window.
        """
        self.root = root
        self.root.title("Hi-STIFS Hardware Control")
        self.root.geometry("400x700")

        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        self.pages = {}
        self.create_home_page()
        self.create_collect_data_page()
        self.create_calibrate_page()
        self.create_process_data_page()
        self.show_page("Home")

        self.status_queue = Queue()
        self.collection_process = None
        self.calibration_data = []

        # Initialize last_config with default values (for running first time on new computer)
        self.last_config = {
            "configuration": "Config 1",
            "pvc_stiffness": "med",
            "height": 80.645,
            "yaw": 5,
            "pitch": 0,
            "roll": 0,
            "rate_of_travel": 25,
            "angle_of_travel": 0,
            "offset_distance": 25,
        }
        self.load_config()

    def load_config(self):
        """Load last_config from config.json if it exists."""
        config_path = os.path.join('AllInOne', 'config.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    for key, value in loaded_config.items():
                        if key in self.last_config:
                            type_func = next(field[2] for field in self.header_fields if field[1] == key)
                            self.last_config[key] = type_func(value)
        except Exception as e:
            self.status_var.set(f"Error loading config: {str(e)}") if hasattr(self, 'status_var') else None

    def save_config(self):
        """Save last_config to config.json."""
        config_path = os.path.join('AllInOne', 'config.json')
        try:
            os.makedirs('AllInOne', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.last_config, f, indent=4)
        except Exception as e:
            self.status_var.set(f"Error saving config: {str(e)}")

    def create_home_page(self):
        """Create the home page with navigation buttons."""
        page = ttk.Frame(self.container)
        self.pages["Home"] = page

        ttk.Label(page, text="Welcome to Hi-STIFS Control", font=("Arial", 16)).pack(pady=20)
        ttk.Button(page, text="Collect Data", command=lambda: self.show_page("Collect Data")).pack(pady=10)
        ttk.Button(page, text="Calibrate", command=lambda: self.show_page("Calibrate")).pack(pady=10)
        ttk.Button(page, text="Process Data", command=lambda: self.show_page("Process Data")).pack(pady=10)

    def create_collect_data_page(self):
        """Create the data collection page with controls."""
        page = ttk.Frame(self.container)
        self.pages["Collect Data"] = page

        ttk.Label(page, text="Arduino Connection", font=("Arial", 14)).pack(pady=10)
        self.connect_button = ttk.Button(page, text="Connect to Arduino", command=self.connect_arduino)
        self.connect_button.pack(pady=5)

        self.available_ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_var = tk.StringVar()
        self.port_combobox = ttk.Combobox(page, textvariable=self.port_var, values=self.available_ports, state="readonly")
        self.port_combobox.pack(pady=5)
        if self.available_ports:
            self.port_var.set(self.available_ports[0])

        ttk.Label(page, text="Date (MM_DD)", font=("Arial", 10)).pack(pady=5)
        self.month_date_var = tk.StringVar(value=datetime.now().strftime("%m_%d"))
        self.month_date_entry = ttk.Entry(page, textvariable=self.month_date_var, width=10)
        self.month_date_entry.pack(pady=5)

        ttk.Label(page, text="Test Number", font=("Arial", 10)).pack(pady=5)
        self.test_num_var = tk.StringVar(value="1")
        self.test_num_entry = ttk.Entry(page, textvariable=self.test_num_var, width=10)
        self.test_num_entry.pack(pady=5)

        self.edit_header_button = ttk.Button(page, text="Edit Header", command=self.edit_header)
        self.edit_header_button.pack(pady=5)

        self.start_button = ttk.Button(page, text="Start Data Collection", command=self.start_collection, state="disabled")
        self.start_button.pack(pady=5)

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(page, textvariable=self.status_var).pack(pady=10)

        ttk.Button(page, text="Home", command=lambda: self.show_page("Home")).pack(pady=10)

    def edit_header(self):
        """Open a window to edit header fields."""
        header_window = tk.Toplevel(self.root)
        header_window.title("Edit Header")
        header_window.geometry("400x700")

        self.config_vars = {}
        for label, key, _ in self.header_fields:
            ttk.Label(header_window, text=label).pack(pady=5)
            var = tk.StringVar(value=str(self.last_config[key]))
            entry = ttk.Entry(header_window, textvariable=var, width=40)
            entry.pack(pady=5)
            self.config_vars[key] = var

        save_button = ttk.Button(header_window, text="Save", command=lambda: self.save_header(header_window))
        save_button.pack(pady=10)

    def save_header(self, window):
        """Save the edited header fields to last_config and config.json."""
        for label, key, type_func in self.header_fields:
            value = self.config_vars[key].get()
            try:
                self.last_config[key] = type_func(value)
            except ValueError:
                self.status_var.set(f"Invalid value for {label}: must be a {type_func.__name__}")
                return
        self.save_config()
        window.destroy()
        self.status_var.set("Header updated")

    def create_calibrate_page(self):
        """Create the calibrate page with controls."""
        page = ttk.Frame(self.container)
        self.pages["Calibrate"] = page

        ttk.Label(page, text="Arduino Connection", font=("Arial", 14)).pack(pady=10)
        self.cal_connect_button = ttk.Button(page, text="Connect to Arduino", command=self.connect_arduino_cal)
        self.cal_connect_button.pack(pady=5)

        self.cal_available_ports = [port.device for port in serial.tools.list_ports.comports()]
        self.cal_port_var = tk.StringVar()
        self.cal_port_combobox = ttk.Combobox(page, textvariable=self.cal_port_var, values=self.cal_available_ports, state="readonly")
        self.cal_port_combobox.pack(pady=5)
        if self.cal_available_ports:
            self.cal_port_var.set(self.cal_available_ports[0])

        ttk.Label(page, text="Date (MM_DD)", font=("Arial", 10)).pack(pady=5)
        self.cal_month_date_var = tk.StringVar(value=datetime.now().strftime("%m_%d"))
        self.cal_month_date_entry = ttk.Entry(page, textvariable=self.cal_month_date_var, width=10)
        self.cal_month_date_entry.pack(pady=5)

        ttk.Label(page, text="Masses (grams, space-separated)", font=("Arial", 10)).pack(pady=5)
        self.masses_var = tk.StringVar()
        self.masses_entry = ttk.Entry(page, textvariable=self.masses_var, width=20)
        self.masses_entry.pack(pady=5)

        ttk.Label(page, text="Positions (cm, space-separated)", font=("Arial", 10)).pack(pady=5)
        self.positions_var = tk.StringVar()
        self.positions_entry = ttk.Entry(page, textvariable=self.positions_var, width=20)
        self.positions_entry.pack(pady=5)

        self.calibrate_button = ttk.Button(page, text="Start Strain Calibration", command=self.start_calibration, state="disabled")
        self.calibrate_button.pack(pady=5)

        self.calibrate_accel_button = ttk.Button(page, text="Start Accel Calibration", command=self.start_accel_calibration, state="disabled")
        self.calibrate_accel_button.pack(pady=5)

        self.load_summary_button = ttk.Button(page, text="Load Summary CSV", command=self.load_summary_csv)
        self.load_summary_button.pack(pady=5)

        self.calc_coefficients_button = ttk.Button(page, text="Calculate Coefficients", command=self.calculate_coefficients)
        self.calc_coefficients_button.pack(pady=5)

        self.cal_status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(page, textvariable=self.cal_status_var).pack(pady=10)

        ttk.Button(page, text="Home", command=lambda: self.show_page("Home")).pack(pady=10)

    def create_process_data_page(self):
        """Create the process data page with controls."""
        page = ttk.Frame(self.container)
        self.pages["Process Data"] = page

        ttk.Label(page, text="Process Data", font=("Arial", 14)).pack(pady=10)

        ttk.Label(page, text="Date (MM_DD)", font=("Arial", 10)).pack(pady=5)
        self.process_month_date_var = tk.StringVar(value=datetime.now().strftime("%m_%d"))
        ttk.Entry(page, textvariable=self.process_month_date_var, width=10).pack(pady=5)

        ttk.Label(page, text="Test Number", font=("Arial", 10)).pack(pady=5)
        self.process_test_num_var = tk.StringVar(value="1")
        ttk.Entry(page, textvariable=self.process_test_num_var, width=10).pack(pady=5)

        ttk.Button(page, text="Process Data", command=self.start_processing).pack(pady=5)

        self.process_status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(page, textvariable=self.process_status_var).pack(pady=10)

        ttk.Button(page, text="Home", command=lambda: self.show_page("Home")).pack(pady=10)

    def show_page(self, page_name):
        """Display the specified page.

        Args:
            page_name (str): Name of the page to show.
        """
        for page in self.pages.values():
            page.pack_forget()
        if page_name in self.pages:
            self.pages[page_name].pack(fill="both", expand=True)
        if page_name == "Collect Data":
            self.available_ports = [port.device for port in serial.tools.list_ports.comports()]
            self.port_combobox["values"] = self.available_ports
            if self.available_ports:
                self.port_var.set(self.available_ports[0])
                self.connect_button["state"] = "normal"
            else:
                self.status_var.set("Status: No COM ports found")
                self.connect_button["state"] = "disabled"
            self.start_button["state"] = "disabled"
            self.month_date_var.set(datetime.now().strftime("%m_%d"))
            self.test_num_var.set("1")
        elif page_name == "Calibrate":

            self.cal_available_ports = [port.device for port in serial.tools.list_ports.comports()]
            self.cal_port_combobox["values"] = self.cal_available_ports
            if self.cal_available_ports:
                self.cal_port_var.set(self.cal_available_ports[0])
                self.cal_connect_button["state"] = "normal"
            else:
                self.cal_status_var.set("Status: No COM ports found")
                self.cal_connect_button["state"] = "disabled"
            self.calibrate_button["state"] = "disabled"
            self.cal_month_date_var.set(datetime.now().strftime("%m_%d"))
        elif page_name == "Process Data":
            self.process_month_date_var.set(datetime.now().strftime("%m_%d"))
            self.process_test_num_var.set("1")
            self.process_status_var.set("Status: Idle")

    def connect_arduino(self):
        """Enable data collection after selecting a port."""
        self.connect_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.port_var.get()
        if not selected_port:
            self.status_var.set("Status: No COM port selected")
            self.connect_button["state"] = "normal"
            return
        self.start_button["state"] = "normal"
        self.status_var.set(f"Status: Ready to collect data from {selected_port}")

    def connect_arduino_cal(self):
        """Enable calibration after selecting a port."""
        self.cal_connect_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.cal_port_var.get()
        if not selected_port:
            self.cal_status_var.set("Status: No COM port selected")
            self.cal_connect_button["state"] = "normal"
            return
        self.calibrate_button["state"] = "normal"
        self.calibrate_accel_button["state"] = "normal"
        self.cal_status_var.set(f"Status: Ready to calibrate from {selected_port}")

    def start_collection(self):
        """Start data collection in a separate process."""
        self.start_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.port_var.get()
        month_date = self.month_date_var.get()
        test_num = self.test_num_var.get()

        if not month_date or not test_num:
            self.status_var.set("Status: Please enter date and test number")
            self.start_button["state"] = "normal"
            return
        if not month_date.replace("_", "").isdigit() or len(month_date) != 5 or month_date[2] != "_":
            self.status_var.set("Status: Invalid date format (use MM_DD)")
            self.start_button["state"] = "normal"
            return
        if not test_num.isdigit():
            self.status_var.set("Status: Test number must be a number")
            self.start_button["state"] = "normal"
            return

        config = {
            "date": month_date,
            "test_num": int(test_num),
        }
        config.update(self.last_config)
        self.collection_process = Process(target=run_collection, args=(selected_port, config, self.status_queue))
        self.collection_process.start()
        self.check_status_queue()

    def start_calibration(self):
        """Start calibration in a separate process."""
        self.calibrate_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.cal_port_var.get()
        month_date = self.cal_month_date_var.get()
        masses = self.masses_var.get()
        positions = self.positions_var.get()

        if not month_date or not masses or not positions:
            self.cal_status_var.set("Status: Please enter all fields")
            self.cal_connect_button["state"] = "normal"
            self.calibrate_button["state"] = "normal"
            return
        if not month_date.replace("_", "").isdigit() or len(month_date) != 5 or month_date[2] != "_":
            self.cal_status_var.set("Status: Invalid date format (use MM_DD)")
            self.cal_connect_button["state"] = "normal"
            self.calibrate_button["state"] = "normal"
            return

        try:
            masses_list = [float(m) for m in masses.split()]
            positions_list = [float(p) for p in positions.split()]
            if not masses_list or not positions_list:
                raise ValueError
        except ValueError:
            self.cal_status_var.set("Status: Masses and positions must be numbers")
            self.cal_connect_button["state"] = "normal"
            self.calibrate_button["state"] = "normal"
            return

        config = {
            "date": month_date,
            "masses": masses_list,
            "positions": positions_list,
            "configuration": "Config 1",
            "pvc_stiffness": "Med",
            "height": 80.645,
            "yaw": 5,
            "pitch": 0,
            "roll": 0,
            "rate_of_travel": 25,
            "angle_of_travel": 0,
            "offset_distance": 25,
        }
        self.collection_process = Process(target=run_calibration, args=(selected_port, config, self.status_queue))
        self.collection_process.start()
        self.check_status_queue()

    def start_accel_calibration(self):
        """Start accel calibration in a separate process."""
        self.calibrate_accel_button["state"] = "disabled"
        self.root.update()
        time.sleep(1)
        selected_port = self.cal_port_var.get()
        month_date = self.cal_month_date_var.get()

        config = {
            "date": month_date,
        }
        self.collection_process = Process(target=run_accel_calibration, args=(selected_port, config, self.status_queue))
        self.collection_process.start()
        self.check_status_queue()

    def load_summary_csv(self):
        """Open a file dialog to select and read a calibration summary CSV."""
        file_path = filedialog.askopenfilename(
            title="Select Calibration Summary CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            self.cal_status_var.set("Status: No file selected")
            return

        try:
            with open(file_path, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader, None)  # Skip header
                headers = next(csvreader, None)
                if headers != ['Mass (g)', 'Position (cm)', 'Avg Strain A1', 'Avg Strain A2', 'Avg Strain B1', 'Avg Strain B2']:
                    self.cal_status_var.set("Status: Invalid CSV format")
                    return
                self.calibration_data = [row for row in csvreader if row]
                summary = f"Loaded {len(self.calibration_data)} records from {os.path.basename(file_path)}"
                self.cal_status_var.set(f"Status: {summary}")
        except Exception as e:
            self.cal_status_var.set(f"Status: Error reading file: {str(e)}")

    def calculate_coefficients(self):
        """Calculate calibration coefficients using linear regression from calibration module."""
        if not self.calibration_data:
            self.cal_status_var.set("Status: No calibration data loaded")
            return
        self.calibration_coeff = calculate_coefficients(self.calibration_data, self.cal_status_var)
        self.cal_status_var.set(f"Status: Coefficients calculated:\n{self.calibration_coeff}")

        # Create AllInOne folder
        parent_folder = 'AllInOne'
        os.makedirs(parent_folder, exist_ok=True)

        # Path for calibration history CSV
        csv_path = os.path.join(parent_folder, 'calibration_history.csv')

        # Parse coefficients
        coeff_lines = self.calibration_coeff.split('\n')
        coeff_data = {}
        for line in coeff_lines:
            if line:
                sensor, values = line.split(': ')
                k, d, c = [float(v.split('=')[1]) for v in values.split(', ')]
                coeff_data[sensor] = {'k': k, 'd': d, 'c': c}

        # Prepare row data
        row = [
            datetime.now().strftime("%m_%d_%Y"),
            coeff_data['A1']['k'], coeff_data['B1']['k'], coeff_data['A2']['k'], coeff_data['B2']['k'],
            coeff_data['A1']['d'], coeff_data['B1']['d'], coeff_data['A2']['d'], coeff_data['B2']['d'],
            coeff_data['A1']['c'], coeff_data['B1']['c'], coeff_data['A2']['c'], coeff_data['B2']['c']
        ]

        # Write to CSV
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                headers = ['Date', 'k_A1', 'k_A2', 'k_B1', 'k_B2', 'd_A1', 'd_A2', 'd_B1', 'd_B2', 'c_A1', 'c_A2', 'c_B1', 'c_B2']
                csvwriter.writerow(headers)
            csvwriter.writerow(row)

    def start_processing(self):
        """Start data processing in a separate process."""
        month_date = self.process_month_date_var.get()
        test_num = self.process_test_num_var.get()

        if not month_date or not test_num:
            self.process_status_var.set("Status: Please enter date and test number")
            return
        if not month_date.replace("_", "").isdigit() or len(month_date) != 5 or month_date[2] != "_":
            self.process_status_var.set("Status: Invalid date format (use MM_DD)")
            return
        if not test_num.isdigit():
            self.process_status_var.set("Status: Test number must be a number")
            return

        self.collection_process = Process(target=show_force_position, args=([month_date], [int(test_num)]))
        self.collection_process.start()
        self.process_status_var.set("Status: Processing started")

    def reset_collect_data_page(self):
        """Reset the Collect Data page to initial state for a new test."""
        self.connect_button["state"] = "normal" if self.available_ports else "disabled"
        self.start_button["state"] = "normal" if self.port_var.get() else "disabled"
        self.month_date_var.set(datetime.now().strftime("%m_%d"))
        self.test_num_var.set(str(int(self.test_num_var.get()) + 1))
        self.status_var.set("Status: Ready for new test")
        self.collection_process = None

    def reset_calibrate_page(self):
        """Reset the Calibrate page to initial state for a new calibration."""
        self.cal_connect_button["state"] = "normal" if self.cal_available_ports else "disabled"
        self.calibrate_button["state"] = "normal" if self.cal_port_var.get() else "disabled"
        self.calibrate_accel_button["state"] = "normal" if self.cal_port_var.get() else "disabled"
        self.cal_month_date_var.set(datetime.now().strftime("%m_%d"))
        self.masses_var.set("")
        self.positions_var.set("")
        self.cal_status_var.set("Status: Ready for new calibration")
        self.collection_process = None

    def check_status_queue(self):
        """Periodically check the status queue and update the UI."""
        while not self.status_queue.empty():
            message = self.status_queue.get()
            if self.pages["Collect Data"].winfo_ismapped():
                self.status_var.set(f"Status: {message}")
                if message == "Data collection ended":
                    self.reset_collect_data_page()
            elif self.pages["Calibrate"].winfo_ismapped():
                self.cal_status_var.set(f"Status: {message}")
                if message == "Calibration ended":
                    self.reset_calibrate_page()
        self.root.after(100, self.check_status_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = HardwareControlUI(root)
    root.mainloop()