import serial
import csv
import keyboard
import numpy as np
import pandas as pd
from datetime import datetime
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from multiprocessing import Queue
import time
import os

# Constants
SUPPLY_VOLTAGE = 5
RESOLUTION = 2**9
GAIN = 16

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
        time_sec (np.ndarray): Array of time values.
        strain_a1 (np.ndarray): Array of A1 strain values.
        strain_b1 (np.ndarray): Array of B1 strain values.
        strain_a2 (np.ndarray): Array of A2 strain values.
        strain_b2 (np.ndarray): Array of B2 strain values.
        force (np.ndarray): Array of force values.
        position (np.ndarray): Array of position values.
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
            self.k_B1 = latest_cal['k_B1']
            self.k_2 = latest_cal['k_A2']
            self.k_B2 = latest_cal['k_B2']
            self.d_1 = latest_cal['d_A1']
            self.d_B1 = latest_cal['d_B1']
            self.d_2 = latest_cal['d_A2']
            self.d_B2 = latest_cal['d_B2']
            self.c_1 = latest_cal['c_A1']
            self.c_B1 = latest_cal['c_B1']
            self.c_2 = latest_cal['c_A2']
            self.c_B2 = latest_cal['c_B2']
            self.m_x = latest_acc_cal['Gain X']
            self.b_x = latest_acc_cal['Offset X']
            self.m_y = latest_acc_cal['Gain Y']
            self.b_y = latest_acc_cal['Offset Y']
            self.m_z = latest_acc_cal['Gain Z']
            self.b_z = latest_acc_cal['Offset Z']
        except Exception as e:
            self.status_queue.put(f"Error loading calibration: {str(e)}")
            self.ser.close()
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
            ["rodney configuration", config["configuration"], '', '', '', ''],
            ["sensor calibration (k d c)", f'{self.k_1} {self.k_B1} {self.k_2} {self.k_B2}', f'{self.d_1} {self.d_B1} {self.d_2} {self.d_B2}', f'{self.c_1} {self.c_B1} {self.c_2} {self.c_B2}', '', ''],
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

        pg.setConfigOptions(antialias=True)
        # Strain plot window
        self.win_strain = pg.GraphicsLayoutWidget(show=True, title=f"Strain Data Plots - Test {config['test_num']}")
        self.win_strain.resize(1000, 500)
        self.win_strain.move(0, 0)
        self.plot_1 = self.win_strain.addPlot(title='1 Strain')
        self.curve_1 = self.plot_1.plot(pen='r', name='1 Strain')
        self.curve_b1 = self.plot_1.plot(pen='b', name='B1 Strain')
        self.plot_2 = self.win_strain.addPlot(title='2 Strain')
        self.curve_2 = self.plot_2.plot(pen='r', name='2 Strain')
        self.curve_b2 = self.plot_2.plot(pen='b', name='B1 Strain')

        # Force and position plot window
        self.win_force_pos = pg.GraphicsLayoutWidget(show=True, title=f"Force and Position - Test {config['test_num']}")
        self.win_force_pos.resize(1000, 500)
        self.win_force_pos.move(1000, 0)
        self.plot_force = self.win_force_pos.addPlot(title='Force')
        self.curve_force = self.plot_force.plot(pen='g', name='Force')
        self.plot_pos = self.win_force_pos.addPlot(title='Position')
        self.curve_pos = self.plot_pos.plot(pen='y', name='Position')

        # Accel plot window
        self.win_accel = pg.GraphicsLayoutWidget(show=True, title=f"Accelerometer Data - Test {config['test_num']}")
        self.win_accel.resize(1000, 500)
        self.win_accel.move(0, 600)
        self.plot_mpuXY = self.win_accel.addPlot(title='Pitch (red) & Roll (green)')
        self.curve_pitch = self.plot_mpuXY.plot(pen='r', name='X')
        self.curve_roll = self.plot_mpuXY.plot(pen='g', name='Y')
        
        self.plot_mpuZ = self.win_accel.addPlot(title='Z Acceleration (g)')
        self.curve_z1 = self.plot_mpuZ.plot(pen='b', name='Z')

        self.time_sec = []
        self.strain_1 = []
        self.strain_2 = []
        self.strain_b1 = []
        self.strain_b2 = []
        self.force = []
        self.position = []
        self.pitch = []
        self.roll = []
        self.acz1 = []

        self.time_offset_check = True
        self.time_offset = 0
        self.plot_time = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

    def update_plot(self):
        """Update the strain, force, and position plots with new data from the serial port."""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            if line == "test ended" or keyboard.is_pressed('space'):
                self.status_queue.put("Data collection ended")
                self.ser.close()
                self.csvfile.close()
                self.timer.stop()
                self.win_strain.close()
                self.win_force_pos.close()
                self.win_accel.close()
                return

            if line.startswith('$'):
                self.status_queue.put(f"Reset at {float(line.split(',')[1])*1e-6}")
                return

            data = line.split(',')
            if len(data) != 11-3-2:
                self.status_queue.put("Invalid data packet")
                

            try:
                time_sec = float(data[0]) * 1e-6
                strain_1 = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                strain_2 = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                strain_b1 =0# float(data[0]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                strain_b2 =0# float(data[0]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                acx1 =0# float(data[3])
                acy1 =0# float(data[4])
                acz1 =0# float(data[5])
            except (ValueError, IndexError):
                self.status_queue.put("Cannot parse data")
                return

            if self.time_offset_check:
                self.time_offset = time_sec
                self.time_offset_check = False
            time_sec -= self.time_offset

            now = datetime.now()
            self.csvwriter.writerow([time_sec, strain_1, strain_2, strain_b1, strain_b2, acx1, acy1, acz1, now.time()])
            self.csvfile.flush()

            # Calculate force and position and angles
            force = (self.k_2 * (strain_1 - self.c_1) - self.k_1 * (strain_2 - self.c_2)) / (self.k_1 * self.k_2 * (self.d_2 - self.d_1))
            position = (self.k_2 * self.d_2 * (strain_1 - self.c_1) - self.k_1 * self.d_1 * (strain_2 - self.c_2)) / (self.k_2 * (strain_1 - self.c_1) - self.k_1 * (strain_2 - self.c_2))
            position = 0 if position > 0.30 or position < -0.30 else position

            # Calculate pitch and roll (in radians) 
            x_g = acx1*self.m_x + self.b_x
            y_g = acy1*self.m_y + self.b_y
            z_g = acz1*self.m_z + self.b_z
            theta_x = np.arctan2(-y_g, np.sqrt(x_g**2 + z_g**2))  # Angle about global x-axis
            theta_y = np.arctan2(x_g, np.sqrt(y_g**2 + z_g**2))  # Angle about global y-axis

            x_angle = np.degrees(theta_x)
            y_angle = np.degrees(theta_y)

            increment = 0.05
            if time_sec - self.plot_time > increment:
                self.time_sec.append(time_sec)
                self.strain_1.append(strain_1)
                self.strain_2.append(strain_2)
                self.strain_b1.append(strain_b1)
                self.strain_b2.append(strain_b2)
                self.force.append(force)
                self.position.append(position * 100)
                self.pitch.append(x_angle)
                self.roll.append(y_angle)
                self.acz1.append(z_g)

                self.curve_1.setData(self.time_sec, self.strain_1)
                self.curve_2.setData(self.time_sec, self.strain_2)
                self.curve_b1.setData(self.time_sec, self.strain_b1)
                self.curve_b2.setData(self.time_sec, self.strain_b2)
                self.curve_force.setData(self.time_sec, self.force)
                self.curve_pos.setData(self.time_sec, self.position)
                self.curve_pitch.setData(self.time_sec, self.pitch)
                self.curve_roll.setData(self.time_sec, self.roll)
                self.curve_z1.setData(self.time_sec, self.acz1)

                x_min = max(0, self.time_sec[-1] - 10)
                x_max = self.time_sec[-1]
                self.plot_1.setXRange(x_min, x_max)
                self.plot_2.setXRange(x_min, x_max)
                self.plot_force.setXRange(x_min, x_max)
                self.plot_pos.setXRange(x_min, x_max)
                self.plot_mpuXY.setXRange(x_min, x_max)
                self.plot_mpuZ.setXRange(x_min, x_max)

                self.plot_time = time_sec

            self.status_queue.put(f"Press 'space' to end data collection")

        except KeyboardInterrupt:
            self.status_queue.put("Interrupted by user")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win_strain.close()
            self.win_force_pos.close()
        except Exception as e:
            self.status_queue.put(f"Error: {str(e)}")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win_strain.close()
            self.win_force_pos.close()

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

