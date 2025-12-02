import serial
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QWidget
from multiprocessing import Queue
import time
import os
import collections
import keyboard

# === ADS1115 parameters ===
ADS1115_BITS            = 15                    # True resolution in differential mode (signed)
ADS1115_MAX_COUNT       = 2**ADS1115_BITS       # 32768 (from –32768 to +32767)
ADS1115_PGA_GAIN        = 16                    # Your configured gain
ADS1115_FSR_AT_GAIN_1   = 4.096                 # Full-scale range at gain ×1 (±4.096 V)
SUPPLY_VOLTAGE          = 5.0                   # Bridge excitation (kept only for documentation/clarity)

# Full-scale range at your actual gain ×16
FSR_AT_CURRENT_GAIN = ADS1115_FSR_AT_GAIN_1 / ADS1115_PGA_GAIN   # ±0.256 V

# Total number of ADC counts across the full ±FSR range
TOTAL_COUNTS_AT_GAIN = ADS1115_MAX_COUNT * ADS1115_PGA_GAIN     # 32768 × 16 = 524288

# Volts per LSB (least significant bit from ADC)
VOLTS_PER_LSB = (2 * FSR_AT_CURRENT_GAIN) / TOTAL_COUNTS_AT_GAIN
# → 0.512 V / 524288 = 1 / 131072 V ≈ 7.62939 µV/LSB

# NOTE ON UNITS:
# - CSV file: all strain values are stored in Volts (required for calibration consistency)
# - Live plots: strain traces are displayed in millivolts (×1000) for readability
# - Force/position calculations use the original Volt values — never altered

# === Plotting parameters ===
PLOT_REFRESH_HZ = 120  # Refresh rate for plot updates in Hz; start at 60, test up to 120 if stable

class SerialReader(QtCore.QThread):
    """Thread for reading serial data, processing, and writing to CSV."""

    data_ready = QtCore.pyqtSignal(list)  # Emits processed data for plotting
    stop_signal = QtCore.pyqtSignal()     # Emits to trigger stop in GUI
    status_signal = QtCore.pyqtSignal(str)  # For status messages
    rate_updated = QtCore.pyqtSignal(float)  # Emits updated input rate in Hz

    def __init__(self, ser, csvfile, csvwriter, k_1, d_1, c_1, k_2, d_2, c_2,
                 k_B1, d_B1, c_B1, k_B2, d_B2, c_B2,
                 m_x, b_x, m_y, b_y, m_z, b_z,
                 accelerations_flag, two_sensors_flag):
        super().__init__()
        self.ser = ser
        self.csvfile = csvfile
        self.csvwriter = csvwriter
        self.k_1 = k_1
        self.d_1 = d_1
        self.c_1 = c_1
        self.k_2 = k_2
        self.d_2 = d_2
        self.c_2 = c_2
        self.k_B1 = k_B1
        self.d_B1 = d_B1
        self.c_B1 = c_B1
        self.k_B2 = k_B2
        self.d_B2 = d_B2
        self.c_B2 = c_B2
        self.m_x = m_x
        self.b_x = b_x
        self.m_y = m_y
        self.b_y = b_y
        self.m_z = m_z
        self.b_z = b_z
        self.accelerations_flag = accelerations_flag
        self.two_sensors_flag = two_sensors_flag
        self.running = True
        self.packet_times = collections.deque(maxlen=10000)  # Timestamps of received packets
        self.last_rate_time = time.time()

    def run(self):
        time_offset_check = True
        time_offset = 0.0
        while self.running:
            if keyboard.is_pressed('space'):
                self.stop_signal.emit()
                break

            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line == "test ended":
                self.stop_signal.emit()
                break
            if line.startswith('$'):
                reset_time = float(line.split(',')[1]) * 1e-6 if len(line.split(',')) > 1 else 0.0
                self.status_signal.emit(f"Reset at {reset_time}")
                continue
            data = line.split(',')
            expected_len = 5 if not self.accelerations_flag else 8  # Adjust if accelerations enabled
            if len(data) != expected_len:
                self.status_signal.emit("Invalid data packet")
                continue
            try:
                time_sec = float(data[0]) * 1e-6
                strain_1 = float(data[1]) * VOLTS_PER_LSB
                strain_2 = float(data[2]) * VOLTS_PER_LSB
                strain_B1 = float(data[3]) * VOLTS_PER_LSB
                strain_B2 = float(data[4]) * VOLTS_PER_LSB
                acx1 = 0.0
                acy1 = 0.0
                acz1 = 0.0
                if self.accelerations_flag:
                    acx1 = float(data[5])
                    acy1 = float(data[6])
                    acz1 = float(data[7])
            except (ValueError, IndexError):
                self.status_signal.emit("Cannot parse data")
                continue

            if time_offset_check:
                time_offset = time_sec
                time_offset_check = False
            time_sec -= time_offset

            self.packet_times.append(time.time())  # Record packet arrival time

            now = datetime.now()
            self.csvwriter.writerow([time_sec, strain_1, strain_2, strain_B1, strain_B2, acx1, acy1, acz1, now.time()])
            self.csvfile.flush()

            # Calculate force and position
            force = (self.k_2 * (strain_1 - self.c_1) - self.k_1 * (strain_2 - self.c_2)) / (self.k_1 * self.k_2 * (self.d_2 - self.d_1))
            num = (self.k_2 * self.d_2 * (strain_1 - self.c_1) - self.k_1 * self.d_1 * (strain_2 - self.c_2))
            den = (self.k_2 * (strain_1 - self.c_1) - self.k_1 * (strain_2 - self.c_2))
            position = num / den if abs(den) > 2.5e-5 else 0.0
            position = 0.0 if position > 0.25 or position < -0.10 else position

            force_B = 0.0
            position_B = 0.0
            if self.two_sensors_flag:
                force_B = (self.k_B2 * (strain_B1 - self.c_B1) - self.k_B1 * (strain_B2 - self.c_B2)) / (self.k_B1 * self.k_B2 * (self.d_B2 - self.d_B1))
                num_B = (self.k_B2 * self.d_B2 * (strain_B1 - self.c_B1) - self.k_B1 * self.d_B1 * (strain_B2 - self.c_B2))
                den_B = (self.k_B2 * (strain_B1 - self.c_B1) - self.k_B1 * (strain_B2 - self.c_B2))
                position_B = num_B / den_B if abs(den_B) > 2.5e-5 else 0.0
                position_B = 0.0 if position_B > 0.25 or position_B < -0.10 else position_B

            pitch = 0.0
            roll = 0.0
            z_g = 0.0
            if self.accelerations_flag:
                x_g = acx1 * self.m_x + self.b_x
                y_g = acy1 * self.m_y + self.b_y
                z_g = acz1 * self.m_z + self.b_z
                theta_x = np.arctan2(-y_g, np.sqrt(x_g**2 + z_g**2))
                theta_y = np.arctan2(x_g, np.sqrt(y_g**2 + z_g**2))
                pitch = np.degrees(theta_x)
                roll = np.degrees(theta_y)

            # Emit data for plotting
            self.data_ready.emit([time_sec, strain_1, strain_2, strain_B1, strain_B2,
                                  force, position * 100, force_B, position_B * 100,
                                  pitch, roll, z_g])

            # Update input rate periodically
            current_time = time.time()
            if current_time - self.last_rate_time > 1.0:
                if self.packet_times:
                    recent_count = sum(1 for t in self.packet_times if current_time - t <= 3.0)
                    rate = recent_count / 3.0
                    self.rate_updated.emit(rate)
                self.last_rate_time = current_time

class RealTimePlotWindow(QtWidgets.QMainWindow):
    """Class to handle real-time strain, force, and position data collection and plotting from an Arduino.

    Attributes:
        ser (serial.Serial): Serial connection to the Arduino.
        csvfile (file): CSV file for data storage.
        csvwriter (csv.writer): Writer for CSV data.
        win_strain (pg.GraphicsLayoutWidget): PyQtGraph window for strain plotting.
        win_force_pos (pg.GraphicsLayoutWidget): PyQtGraph window for force and position plotting.
        plot_x (pg.PlotItem): Plot for A1 & B1 strains.
        plot_y (pg.PlotItem): Plot for A2 & B2 strains.
        plot_force (pg.PlotItem): Plot for force.
        plot_pos (pg.PlotItem): Plot for position.
        curve_a1 (pg.PlotDataItem): Plot curve for A1 strain.
        curve_b1 (pg.PlotDataItem): Plot curve for B1 strain.
        curve_a2 (pg.PlotDataItem): Plot curve for A2 strain.
        curve_b2 (pg.PlotDataItem): Plot curve for B2 strain.
        curve_force (pg.PlotDataItem): Plot curve for force.
        curve_pos (pg.PlotDataItem): Plot curve for position.
        time_sec (collections.deque): Deque of time values.
        strain_a1 (collections.deque): Deque of A1 strain values.
        strain_b1 (collections.deque): Deque of B1 strain values.
        strain_a2 (collections.deque): Deque of A2 strain values.
        strain_b2 (collections.deque): Deque of B2 strain values.
        force (collections.deque): Deque of force values.
        position (collections.deque): Deque of position values.
        time_offset (float): Time offset for data collection.
        time_offset_check (bool): Flag to set initial time offset.
        plot_time (float): Last time plotted to control update frequency.
        timer (QtCore.QTimer): Timer for updating plots.
        k_A1, k_B1, k_A2, k_B2, d_A1, d_B1, d_A2, d_B2, c_A1, c_B1, c_A2, c_B2: Calibration coefficients.
    """

    def __init__(self, port, config, status_queue):
        """Initialize the plot windows and serial connection.

        Args:
            port (str): Serial port for Arduino communication.
            config (dict): Configuration dictionary with test parameters.
            status_queue (Queue): Queue to send status messages to the UI.
        """
        self.accelerations_flag = False
        self.two_sensors_flag = True
        super().__init__()
        self.status_queue = status_queue

        # Load calibration coefficients
        cal_csv_path = r'AllInOne\calibration_history.csv'
        acc_csv_path = r'AllInOne\accel_calibration_history.csv'
        try:
            cal_data = pd.read_csv(cal_csv_path)
            latest_cal = cal_data.iloc[-1]
            acc_cal_data = pd.read_csv(acc_csv_path)
            latest_acc_cal = acc_cal_data.iloc[-1]

            self.k_1 = latest_cal['k_A1']
            self.d_1 = latest_cal['d_A1']
            self.c_1 = latest_cal['c_A1']
            self.k_2 = latest_cal['k_A2']
            self.d_2 = latest_cal['d_A2']
            self.c_2 = latest_cal['c_A2']

            self.k_B1 = latest_cal['k_B1']
            self.d_B1 = latest_cal['d_B1']
            self.c_B1 = latest_cal['c_B1']
            self.k_B2 = latest_cal['k_B2']
            self.d_B2 = latest_cal['d_B2']
            self.c_B2 = latest_cal['c_B2']

            self.m_x = latest_acc_cal['Gain X']
            self.b_x = latest_acc_cal['Offset X']
            self.m_y = latest_acc_cal['Gain Y']
            self.b_y = latest_acc_cal['Offset Y']
            self.m_z = latest_acc_cal['Gain Z']
            self.b_z = latest_acc_cal['Offset Z']
        except Exception as e:
            self.status_queue.put(f"Error loading calibration: {str(e)}")
            return

        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
        except serial.SerialException as e:
            self.status_queue.put(f"Failed to connect to {port}: {str(e)}")
            return

        count = 0
        while True:
            count += 1
            incoming_data = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if count <= 1:
                time.sleep(2)
                self.status_queue.put("Press 'space' to sync with Arduino")
            if incoming_data == "#" or keyboard.is_pressed('space'):
                self.status_queue.put("Starting data collection")
                break

        # Create parent folder based on date
        parent_folder = os.path.join('Raw Data', f'{config["date"]}')
        os.makedirs(parent_folder, exist_ok=True)

        # Open CSV file in the parent folder
        csv_path = os.path.join(parent_folder, f'{config["date"]}_test_{config["test_num"]}.csv')
        self.csvfile = open(csv_path, 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)

        pre_test_notes = [
            ["user_note", '', '', '', '', ''],
            ["configuration", config["configuration"], '', '', '', ''],
            ["sensor calibration (k d c)", f'{self.k_1} {self.k_2} {self.k_B1} {self.k_B2}', f'{self.d_1} {self.d_2} {self.d_B1} {self.d_B2}', f'{self.c_1} {self.c_2} {self.c_B1} {self.c_B2}', '', ''],
            ["stalk array (lo med hi)", config["pvc_stiffness"], '', '', '', ''],
            ["sensor height (cm)", config["height"], '', '', '', ''],
            ["sensor yaw (degrees)", config["yaw"], '', '', '', ''],
            ["sensor pitch (degrees)", config["pitch"], '', '', '', ''],
            ["sensor roll (degrees)", config["roll"], '', '', '', ''],
            ["rate of travel (ft/min)", config["rate_of_travel"], '', '', '', ''],
            ["angle of travel (degrees)", config["angle_of_travel"], '', '', '', ''],
            ["sensor offset (cm to gauge 2)", config["offset_distance"], '', '', '', ''],
            ["====="]
        ]
        for note in pre_test_notes:
            self.csvwriter.writerow(note)

        headers = ['Time', 'Strain A1', 'Strain A2', 'Strain B1', 'Strain B2', 'AcX1', 'AcY1', 'AcZ1', 'Current Time']
        self.csvwriter.writerow(headers)

        # Performance optimizations for high refresh rates
        pg.setConfigOptions(useOpenGL=True, antialias=False)

        # Strain plot window with preset buttons for time range
        self.strain_window = QWidget()
        self.strain_window.setWindowTitle(f"Strain Data Plots - Test {config['test_num']}")
        layout = QVBoxLayout()

        self.win_strain = pg.GraphicsLayoutWidget()
        self.plot_1 = self.win_strain.addPlot(title='Strain 1')
        self.plot_1.setLabel('left', '', units='mV')
        # Display-only scaling: actual data in CSV remains in Volts
        self.curve_1 = self.plot_1.plot(pen='r', name='A1 Strain')
        self.curve_B1 = self.plot_1.plot(pen='b', name='B1 Strain')

        self.plot_2 = self.win_strain.addPlot(title='Strain 2')
        self.plot_2.setLabel('left', '', units='mV')
        # Display-only scaling: actual data in CSV remains in Volts
        self.curve_2 = self.plot_2.plot(pen='r', name='A2 Strain')
        self.curve_B2 = self.plot_2.plot(pen='b', name='B2 Strain')

        layout.addWidget(self.win_strain)

        # Add preset buttons for display time range and rate label
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Time Range (s):")
        preset_layout.addWidget(preset_label)
        presets = [0.1, 0.5, 1, 3, 5, 10, 15, 20]
        for preset in presets:
            btn = QPushButton(str(preset))
            btn.clicked.connect(lambda _, p=preset: self.set_time_range(p))
            preset_layout.addWidget(btn)
        self.rate_label = QLabel("Input Rate: 0 Hz")
        preset_layout.addStretch()
        preset_layout.addWidget(self.rate_label)
        layout.addLayout(preset_layout)

        self.strain_window.setLayout(layout)
        if self.accelerations_flag:
            self.strain_window.resize(1000, 550)  # Slightly taller to fit buttons
            self.strain_window.move(0, 0)
        else:
            self.strain_window.resize(1000, 1050)  # Slightly taller to fit buttons
            self.strain_window.move(0, 0)
        self.strain_window.show()

        self.display_time_range = 10.0  # Initial time range in seconds

        # Force and position plot window
        self.win_force_pos = pg.GraphicsLayoutWidget(show=True, title=f"Force and Position - Test {config['test_num']}")
        if self.accelerations_flag:
            self.win_force_pos.resize(1000, 500)
            self.win_force_pos.move(1000, 0)
        else:
            self.win_force_pos.resize(1000, 1000)
            self.win_force_pos.move(1000, 0)
        self.plot_force = self.win_force_pos.addPlot(title='Force')
        self.curve_force = self.plot_force.plot(pen='r', name='Force')
        if self.two_sensors_flag:
            self.curve_force_B = self.plot_force.plot(pen='b', name='Force B')

        self.plot_pos = self.win_force_pos.addPlot(title='Position')
        self.curve_pos = self.plot_pos.plot(pen='r', name='Position')
        if self.two_sensors_flag:
            self.curve_pos_B = self.plot_pos.plot(pen='b', name='Position B')

        # Accel plot window
        if self.accelerations_flag:
            self.win_accel = pg.GraphicsLayoutWidget(show=True, title=f"Accelerometer Data - Test {config['test_num']}")
            self.win_accel.resize(1000, 500)
            self.win_accel.move(0, 600)
            self.plot_mpuXY = self.win_accel.addPlot(title='Pitch (red) & Roll (green)')
            self.curve_pitch = self.plot_mpuXY.plot(pen='r', name='X')
            self.curve_roll = self.plot_mpuXY.plot(pen='g', name='Y')
            
            self.plot_mpuZ = self.win_accel.addPlot(title='Z Acceleration (g)')
            self.curve_z1 = self.plot_mpuZ.plot(pen='b', name='Z')

        # Use deques for plotting data to limit memory usage
        maxlen = 20 * 120  # Sufficient for ~20s at 120 Hz (safety for high inputs)
        self.time_sec = collections.deque(maxlen=maxlen)
        self.strain_1 = collections.deque(maxlen=maxlen)
        self.strain_2 = collections.deque(maxlen=maxlen)
        self.strain_B1 = collections.deque(maxlen=maxlen)
        self.strain_B2 = collections.deque(maxlen=maxlen)
        self.force = collections.deque(maxlen=maxlen)
        self.position = collections.deque(maxlen=maxlen)
        self.force_B = collections.deque(maxlen=maxlen)
        self.position_B = collections.deque(maxlen=maxlen)
        if self.accelerations_flag:
            self.pitch = collections.deque(maxlen=maxlen)
            self.roll = collections.deque(maxlen=maxlen)
            self.acz1 = collections.deque(maxlen=maxlen)

        # Create and start the serial reader thread
        self.reader = SerialReader(self.ser, self.csvfile, self.csvwriter,
                                   self.k_1, self.d_1, self.c_1, self.k_2, self.d_2, self.c_2,
                                   self.k_B1, self.d_B1, self.c_B1, self.k_B2, self.d_B2, self.c_B2,
                                   self.m_x, self.b_x, self.m_y, self.b_y, self.m_z, self.b_z,
                                   self.accelerations_flag, self.two_sensors_flag)
        self.reader.data_ready.connect(self.handle_data)
        self.reader.stop_signal.connect(self.stop_collection)
        self.reader.status_signal.connect(lambda msg: self.status_queue.put(msg))
        self.reader.rate_updated.connect(lambda rate: self.rate_label.setText(f"Input Rate: {rate:.1f} Hz"))
        self.reader.start()

        # Set up timer for fixed-rate plot updates
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(int(1000 / PLOT_REFRESH_HZ))  # Interval in ms

        self.status_queue.put("Press 'space' to end data collection")

    def handle_data(self, data_list):
        """Handle emitted data from the thread: append to deques (updates handled by timer)."""
        time_sec, strain_1, strain_2, strain_B1, strain_B2, force, position, force_B, position_B, pitch, roll, acz1 = data_list

        self.time_sec.append(time_sec)
        # === Strain values: convert to mV for display only ===
        # NOTE: CSV file still records true voltage in Volts.
        #       Calibration coefficients (k, d, c) were determined using volts,
        #       so we must NOT change the values written to disk.
        self.strain_1.append(strain_1 * 1000.0)   # Display in mV
        self.strain_2.append(strain_2 * 1000.0)   # Display in mV
        self.strain_B1.append(strain_B1 * 1000.0) # Display in mV
        self.strain_B2.append(strain_B2 * 1000.0) # Display in mV
        # ====================================================
        self.force.append(force)
        self.position.append(position)
        self.force_B.append(force_B)
        self.position_B.append(position_B)
        if self.accelerations_flag:
            self.pitch.append(pitch)
            self.roll.append(roll)
            self.acz1.append(acz1)

    def update_plots(self):
        """Update all plot curves and ranges."""
        self.curve_1.setData(self.time_sec, self.strain_1)
        self.curve_2.setData(self.time_sec, self.strain_2)
        self.curve_B1.setData(self.time_sec, self.strain_B1)
        self.curve_B2.setData(self.time_sec, self.strain_B2)
        self.curve_force.setData(self.time_sec, self.force)
        self.curve_pos.setData(self.time_sec, self.position)
        if self.two_sensors_flag:
            self.curve_force_B.setData(self.time_sec, self.force_B)
            self.curve_pos_B.setData(self.time_sec, self.position_B)
        if self.accelerations_flag:
            self.curve_pitch.setData(self.time_sec, self.pitch)
            self.curve_roll.setData(self.time_sec, self.roll)
            self.curve_z1.setData(self.time_sec, self.acz1)

        if self.time_sec:
            x_min = max(0, self.time_sec[-1] - self.display_time_range)
            x_max = self.time_sec[-1]
        else:
            x_min = 0
            x_max = 0

        self.plot_1.setXRange(x_min, x_max)
        self.plot_2.setXRange(x_min, x_max)
        self.plot_force.setXRange(x_min, x_max)
        self.plot_pos.setXRange(x_min, x_max)
        if self.accelerations_flag:
            self.plot_mpuXY.setXRange(x_min, x_max)
            self.plot_mpuZ.setXRange(x_min, x_max)

    def set_time_range(self, value):
        """Set the display time range based on button preset."""
        self.display_time_range = float(value)
        # Trigger an immediate plot update to reflect the new range
        self.update_plots()

    def keyPressEvent(self, event):
        """Handle key press events for stopping collection."""
        if event.key() == QtCore.Qt.Key_Space:
            self.stop_collection()

    def stop_collection(self):
        """Stop data collection and clean up."""
        self.status_queue.put("Data collection ended")
        self.plot_timer.stop()
        self.reader.running = False
        self.reader.wait()  # Wait for thread to finish
        self.ser.close()
        self.csvfile.close()
        self.strain_window.close()
        self.win_force_pos.close()
        if self.accelerations_flag:
            self.win_accel.close()

def run_collection(port, config, status_queue):
    """Run the real-time plot window in a separate process.
    Args:
        port (str): Serial port for Arduino communication.
        config (dict): Configuration dictionary with test parameters.
        status_queue (Queue): Queue to send status messages to the UI.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = RealTimePlotWindow(port, config, status_queue)
    app.exec_()