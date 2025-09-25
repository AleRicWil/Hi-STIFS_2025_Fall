import serial
import csv
import keyboard
import numpy as np
from datetime import datetime
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from multiprocessing import Queue
import time
import os

class RealTimePlotWindow(QtWidgets.QMainWindow):
    """Class to handle real-time accelerometer data collection and plotting from two MPU-6050 sensors via Arduino.

    Attributes:
        ser (serial.Serial): Serial connection to the Arduino.
        csvfile (file): CSV file for data storage.
        csvwriter (csv.writer): Writer for CSV data.
        win_accel (pg.GraphicsLayoutWidget): PyQtGraph window for accelerometer plotting.
        plot_mpu1 (pg.PlotItem): Plot for MPU1 (AcX1, AcY1, AcZ1).
        plot_mpu2 (pg.PlotItem): Plot for MPU2 (AcX2, AcY2, AcZ2).
        curve_x1 (pg.PlotDataItem): Plot curve for AcX1.
        curve_y1 (pg.PlotDataItem): Plot curve for AcY1.
        curve_z1 (pg.PlotDataItem): Plot curve for AcZ1.
        curve_x2 (pg.PlotDataItem): Plot curve for AcX2.
        curve_y2 (pg.PlotDataItem): Plot curve for AcY2.
        curve_z2 (pg.PlotDataItem): Plot curve for AcZ2.
        time_sec (list): List of time values.
        acx1, acy1, acz1, acx2, acy2, acz2 (list): Lists of accelerometer values.
        time_offset (float): Time offset for data collection.
        time_offset_check (bool): Flag to set initial time offset.
        plot_time (float): Last time plotted to control update frequency.
        timer (QtCore.QTimer): Timer for updating plots.
    """

    def __init__(self, port, config, status_queue):
        """Initialize the plot window and serial connection.

        Args:
            port (str): Serial port for Arduino communication.
            config (dict): Configuration dictionary with test parameters.
            status_queue (Queue): Queue to send status messages to the UI.
        """
        super().__init__()
        self.status_queue = status_queue

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
        parent_folder = os.path.join('Raw_Data_Accelerometer', f'{config["date"]}')
        os.makedirs(parent_folder, exist_ok=True)

        # Open CSV file in the parent folder
        csv_path = os.path.join(parent_folder, f'{config["date"]}_test_{config["test_num"]}.csv')
        self.csvfile = open(csv_path, 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)

        pre_test_notes = [
            ["test_note", config["note"], '', '', '', '', ''],
            ["sensor_configuration", config["configuration"], '', '', '', '', ''],
            ["sensor_height_cm", config["height"], '', '', '', '', ''],
            ["sensor_yaw_degrees", config["yaw"], '', '', '', '', ''],
            ["sensor_pitch_degrees", config["pitch"], '', '', '', '', ''],
            ["sensor_roll_degrees", config["roll"], '', '', '', '', ''],
            ["====="]
        ]
        for note in pre_test_notes:
            self.csvwriter.writerow(note)

        headers = ['Time', 'AcX1', 'AcY1', 'AcZ1', 'AcX2', 'AcY2', 'AcZ2', 'Current Time']
        self.csvwriter.writerow(headers)

        pg.setConfigOptions(antialias=True)
        # Accelerometer plot window
        self.win_accel = pg.GraphicsLayoutWidget(show=True, title=f"Accelerometer Data - Test {config['test_num']}")
        self.win_accel.resize(1000, 1000)
        self.win_accel.move(0, 0)
        self.plot_mpu1 = self.win_accel.addPlot(title='MPU1 Acceleration Vs Time')
        self.curve_x1 = self.plot_mpu1.plot(pen='r', name='X')
        self.curve_y1 = self.plot_mpu1.plot(pen='g', name='Y')
        self.curve_z1 = self.plot_mpu1.plot(pen='b', name='Z')
        self.plot_mpu2 = self.win_accel.addPlot(title='MPU2 Acceleration Vs Time')
        self.curve_x2 = self.plot_mpu2.plot(pen='r', name='X')
        self.curve_y2 = self.plot_mpu2.plot(pen='g', name='Y')
        self.curve_z2 = self.plot_mpu2.plot(pen='b', name='Z')

        self.time_sec = []
        self.acx1 = []
        self.acy1 = []
        self.acz1 = []
        self.acx2 = []
        self.acy2 = []
        self.acz2 = []

        self.time_offset_check = True
        self.time_offset = 0
        self.plot_time = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)  # Update every 1ms

    def update_plot(self):
        """Update the accelerometer plots with new data from the serial port."""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            data = line.split(',')

            if len(data) != 6:
                self.status_queue.put("Invalid data received: incomplete data packet")
                return

            if keyboard.is_pressed('space'):
                self.status_queue.put("Data collection ended")
                self.ser.close()
                self.csvfile.close()
                self.timer.stop()
                self.win_accel.close()
                return

            try:
                acx1 = float(data[0])
                acy1 = float(data[1])
                acz1 = float(data[2])
                acx2 = float(data[3])
                acy2 = float(data[4])
                acz2 = float(data[5])
            except (ValueError, IndexError):
                self.status_queue.put("Invalid data received: cannot parse values")
                return

            now = datetime.now()
            current_time = now.time()
            time_sec = (now - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

            if self.time_offset_check:
                self.time_offset = time_sec
                self.time_offset_check = False

            time_sec -= self.time_offset
            self.csvwriter.writerow([time_sec, acx1, acy1, acz1, acx2, acy2, acz2, current_time])
            self.csvfile.flush()

            increment = 0.05  # Update plot every 50ms
            if time_sec - self.plot_time > increment:
                self.time_sec.append(time_sec)
                self.acx1.append(acx1)
                self.acy1.append(acy1)
                self.acz1.append(acz1)
                self.acx2.append(acx2)
                self.acy2.append(acy2)
                self.acz2.append(acz2)

                # Update MPU1 plot
                self.curve_x1.setData(self.time_sec, self.acx1)
                self.curve_y1.setData(self.time_sec, self.acy1)
                self.curve_z1.setData(self.time_sec, self.acz1)

                # Update MPU2 plot
                self.curve_x2.setData(self.time_sec, self.acx2)
                self.curve_y2.setData(self.time_sec, self.acy2)
                self.curve_z2.setData(self.time_sec, self.acz2)

                # Set scrolling window (last 10 seconds)
                if len(self.time_sec) > 0:
                    self.plot_mpu1.setXRange(max(0, self.time_sec[-1] - 10), self.time_sec[-1])
                    self.plot_mpu2.setXRange(max(0, self.time_sec[-1] - 10), self.time_sec[-1])

                self.plot_time = time_sec

            self.status_queue.put(f"Press 'space' to end data collection")

        except KeyboardInterrupt:
            self.status_queue.put("Interrupted by user")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win_accel.close()
        except Exception as e:
            self.status_queue.put(f"Error: {str(e)}")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
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

if __name__ == '__main__':
    # Configuration
    config = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'test_num': 1,
        'note': 'Accelerometer test with two MPU-6050 sensors',
        'configuration': 'Two MPU-6050 on I2C bus',
        'height': 0,
        'yaw': 0,
        'pitch': 0,
        'roll': 0
    }
    status_queue = Queue()
    port = 'COM5 '  # Change to your Arduino's port

    run_collection(port, config, status_queue)