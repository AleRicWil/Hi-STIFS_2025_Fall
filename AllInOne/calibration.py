import serial
import csv
import keyboard
import numpy as np
from datetime import datetime
import time
import os
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# Constants
SUPPLY_VOLTAGE = 5
RESOLUTION = 2**9
GAIN = 16

def run_calibration(port, config, status_queue):
    """Run calibration by collecting strain data for each mass-position pair.

    Args:
        port (str): Serial port for Arduino communication.
        config (dict): Configuration dictionary with calibration parameters.
        status_queue (Queue): Queue to send status messages to the UI.
    """
    parent_folder = os.path.join('Raw Data', f'{config["date"]}')
    os.makedirs(parent_folder, exist_ok=True)
    summary_path = os.path.join(parent_folder, f'{config["date"]}_calibration_summary.csv')
    summary_data = []

    masses = config["masses"]
    positions = config["positions"]

    for mass in masses:
        for position in positions:
            try:
                ser = serial.Serial(port, 115200, timeout=1)
            except serial.SerialException as e:
                status_queue.put(f"Failed to connect to {port}: {str(e)}")
                return

            count = 0
            while True:
                count += 1
                incoming_data = ser.readline().decode('utf-8', errors='ignore').strip()
                if count <= 1:
                    time.sleep(2)
                    status_queue.put(f"Press 'space' to start collection for mass {mass}g at {position}cm")
                if incoming_data == "#" or keyboard.is_pressed('space'):
                    status_queue.put(f"Starting data collection for mass {mass}g at {position}cm")
                    break

            csv_path = os.path.join(parent_folder, f'{config["date"]}_calibration_mass_{mass}_pos_{position}.csv')
            strains = {'a1': [], 'b1': [], 'a2': [], 'b2': []}

            with open(csv_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                pre_test_notes = [
                    ["pre_test_note_1", '', '', '', '', ''],
                    ["mass (g)", mass, '', '', '', ''],
                    ["position (cm)", position, '', '', '', ''],
                    ["=================================================================="]
                ]
                for note in pre_test_notes:
                    csvwriter.writerow(note)

                headers = ['Time', 'Strain A1', 'Strain A2', 'Strain B1', 'Strain B2', 'Current Time']
                csvwriter.writerow(headers)

                time_offset_check = True
                time_offset = 0

                time.sleep(1)
                while True:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        data = line.split(',')

                        if len(data) != 5:
                            status_queue.put("Invalid data received: incomplete data packet")
                            continue

                        if data[0] == "$":
                            status_queue.put(f"Reset at {float(data[1])*10**-6}")
                            continue

                        now = datetime.now()
                        current_time = now.time()

                        if data[0] == "test ended" or keyboard.is_pressed('space'):
                            status_queue.put(f"Data collection ended for mass {mass}g at {position}cm")
                            break

                        if data[0] == " " or data[1] == " ":
                            continue

                        try:
                            time_sec = float(data[0]) * 1e-6
                            strain_1 = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_2 = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_B1 = float(data[3]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_B2 = float(data[4]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            # acx1 = float(data[3])
                            # acy1 = float(data[4])
                            # acz1 = float(data[5])
                        except (ValueError, IndexError):
                            status_queue.put("Invalid data received: cannot parse values")
                            continue

                        if time_offset_check:
                            time_offset = time_sec
                            time_offset_check = False

                        time_sec -= time_offset
                        csvwriter.writerow([time_sec, strain_1, strain_B1, strain_2, strain_B2, current_time])
                        csvfile.flush()

                        strains['a1'].append(strain_1)
                        strains['a2'].append(strain_2)
                        strains['b1'].append(strain_B1)
                        strains['b2'].append(strain_B2)

                        status_queue.put(f"Press 'space' to end data collection for mass {mass}g at {position}cm")

                    except KeyboardInterrupt:
                        status_queue.put("Interrupted by user")
                        ser.close()
                        break
                    except Exception as e:
                        status_queue.put(f"Error: {str(e)}")
                        ser.close()
                        break

                ser.close()

            avg_strains = {
                'a1': np.mean(strains['a1']) if strains['a1'] else 0,
                'a2': np.mean(strains['a2']) if strains['a2'] else 0,
                'b1': np.mean(strains['b1']) if strains['b1'] else 0,
                'b2': np.mean(strains['b2']) if strains['b2'] else 0
            }
            summary_data.append([mass, position, avg_strains['a1'], avg_strains['a2'], avg_strains['b1'], avg_strains['b2']])

    with open(summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        pre_test_notes = [
            ["====="]
        ]
        for note in pre_test_notes:
            csvwriter.writerow(note)

        headers = ['Mass (g)', 'Position (cm)', 'Avg Strain A1', 'Avg Strain A2', 'Avg Strain B1', 'Avg Strain B2']
        csvwriter.writerow(headers)
        for row in summary_data:
            csvwriter.writerow(row)

    status_queue.put("Calibration ended")

def calculate_coefficients(calibration_data, cal_status_var):
    """Calculate calibration coefficients using multiple linear regression to directly fit k, d, and c.

    Args:
        calibration_data (list): List of rows from the calibration summary CSV.
        cal_status_var (tk.StringVar): Tkinter variable to update the UI status.

    Returns:
        str: Formatted string of calculated coefficients.
    """
    if not calibration_data:
        cal_status_var.set("Status: No calibration data loaded")
        return ""

    try:
        # Extract and filter data, ensuring all columns are valid numbers
        valid_data = []
        for row in calibration_data:
            try:
                mass = float(row[0])
                position = float(row[1])
                strain_1 = float(row[2])   #2
                strain_2 = float(row[3])   #3
                strain_B1 = float(row[4])   #4
                strain_B2 = float(row[5])   #5
                valid_data.append((mass, position, strain_1, strain_2, strain_B1, strain_B2))
            except ValueError:
                continue

        if not valid_data:
            cal_status_var.set("Status: No valid data for calculation")
            return ""

        # Unpack filtered data
        masses, positions, strains_1, strains_2, strains_B1, strains_B2 = zip(*valid_data)
        masses = np.array(masses)
        positions = np.array(positions)
        strains_1 = np.array(strains_1)
        strains_2 = np.array(strains_2)
        strains_B1 = np.array(strains_B1)
        strains_B2 = np.array(strains_B2)

        # Match units
        masses = masses * 1e-3  # g to kg
        positions = positions * 1e-2  # cm to m
        g = 9.80  # m/s^2
        forces = masses * g  # Newtons

        # Create design matrix A = [F*x, -F] for multiple linear regression
        A = np.column_stack((forces * positions, -forces))

        # Fit models for each strain type: V = c + k*(F*x) + beta2*(-F), where beta2 = k*d
        model_1 = LinearRegression().fit(A, strains_1)
        c_1 = model_1.intercept_
        k_1 = model_1.coef_[0]
        beta2_1 = model_1.coef_[1]
        d_1 = beta2_1 / k_1 if k_1 != 0 else 0

        model_2 = LinearRegression().fit(A, strains_2)
        c_2 = model_2.intercept_
        k_2 = model_2.coef_[0]
        beta2_2 = model_2.coef_[1]
        d_2 = beta2_2 / k_2 if k_2 != 0 else 0

        model_B1 = LinearRegression().fit(A, strains_B1)
        c_B1 = model_B1.intercept_
        k_B1 = model_B1.coef_[0]
        beta2_B1 = model_B1.coef_[1]
        d_B1 = beta2_B1 / k_B1 if k_B1 != 0 else 0

        model_B2 = LinearRegression().fit(A, strains_B2)
        c_B2 = model_B2.intercept_
        k_B2 = model_B2.coef_[0]
        beta2_B2 = model_B2.coef_[1]
        d_B2 = beta2_B2 / k_B2 if k_B2 != 0 else 0

        # Format result
        result = (f"A1: k={k_1:.6f}, d={d_1:.6f}, c={c_1:.6f}\n"
                  f"A2: k={k_2:.6f}, d={d_2:.6f}, c={c_2:.6f}\n"
                  f"B1: k={k_B1:.6f}, d={d_B1:.6f}, c={c_B1:.6f}\n"
                  f"B2: k={k_B2:.6f}, d={d_B2:.6f}, c={c_B2:.6f}")

        # Optional plotting for A1 strain
        V_1_pred = model_1.predict(A)
        V_2_pred = model_2.predict(A)
        V_B1_pred = model_B1.predict(A)
        V_B2_pred = model_B2.predict(A)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(forces * positions, strains_1, label='Original')
        plt.scatter(forces * positions, V_1_pred, label='Predicted')
        plt.title('Gauge 1')
        plt.legend()

        plt.figure(2)
        plt.scatter(forces * positions, strains_2, label='Original')
        plt.scatter(forces * positions, V_2_pred, label='Predicted')
        plt.legend()
        plt.title('Gauge 2')

        plt.figure(3)
        plt.scatter(forces * positions, strains_B1, label='Original')
        plt.scatter(forces * positions, V_B1_pred, label='Predicted')
        plt.title('Gauge B1')
        plt.legend()

        plt.figure(4)
        plt.scatter(forces * positions, strains_B2, label='Original')
        plt.scatter(forces * positions, V_B2_pred, label='Predicted')
        plt.legend()
        plt.title('Gauge B2')

        plt.show()

        return result

    except Exception as e:
        cal_status_var.set(f"Status: Error calculating coefficients: {str(e)}")
        return ""
    

def compute_accel_calibration(measurements, status_queue, g=1):
    """Compute calibration offsets and gains using least-squares optimization.
    
    Args:
        measurements (np.array): Array of shape (N, 3) with raw [x, y, z] samples for N orientations.
        status_queue (Queue): Queue to send status messages to the UI.
        g (float): Gravity acceleration in m/s² (default: 9.81).
    
    Returns:
        offsets (np.array): Offsets for [x, y, z].
        gains (np.array): Gains for [x, y, z].
    """
    # Validate input data
    if len(measurements) < 6:
        status_queue.put("Error: Fewer than 6 orientations provided.")
        return None, None
    spreads = np.max(measurements, axis=0) - np.min(measurements, axis=0)
    if np.any(spreads < 0.1 * np.max(spreads)):
        status_queue.put("Warning: Measurements lack sufficient variation in one or more axes.")
        # Continue with fallback if optimization fails

    def calibration_error(params, measurements, g):
        offsets = params[:3]
        gains = params[3:]
        error = 0
        for meas in measurements:
            corrected = (meas - offsets) / gains
            magnitude = np.sqrt(np.sum(corrected**2))
            error += (magnitude - g)**2
        return error

    # Improved initial guess
    offsets_guess = np.mean(measurements, axis=0)
    ranges = np.max(measurements, axis=0) - np.min(measurements, axis=0)
    gains_guess = ranges / (2 * g)  # Assume range spans ±1g
    initial_guess = np.concatenate([offsets_guess, gains_guess])

    # Optimize with SLSQP for robustness
    result = minimize(
        calibration_error,
        initial_guess,
        args=(measurements, g),
        method='SLSQP',
        bounds=[(None, None)]*3 + [(0.1, None)]*3,  # Gains must be positive
        options={'maxiter': 1000, 'disp': False}
    )

    if result.success:
        offsets = result.x[:3]
        gains = result.x[3:]
        status_queue.put(f"Optimization successful. Final error: {result.fun:.6f}")
    else:
        status_queue.put("Optimization failed. Using approximate calibration parameters.")
        offsets = offsets_guess
        gains = gains_guess  # Fallback to approximate values

    return offsets, gains

def run_accel_calibration(port, config, status_queue):
    """Run accelerometer calibration by collecting data in six stationary orientations.

    Args:
        port (str): Serial port for Arduino communication.
        config (dict): Configuration dictionary with calibration parameters.
        status_queue (Queue): Queue to send status messages to the UI.
    """
    parent_folder = os.path.join('Raw Data', f'{config["date"]}')
    os.makedirs(parent_folder, exist_ok=True)
    csv_path = os.path.join(parent_folder, f'{config["date"]}_accel_calibration.csv')

    orientations = [
        "X-axis up",
        "X-axis down",
        "Y-axis up",
        "Y-axis down",
        "Z-axis up",
        "Z-axis down"
    ]
    stationary_samples = []

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write pre-test notes
        pre_test_notes = [
            ["pre_test_note_1", '', '', '', '', ''],
            ["====="]
        ]
        for note in pre_test_notes:
            csvwriter.writerow(note)
        headers = ['Time', 'Accel X', 'Accel Y', 'Accel Z', 'Orientation']
        csvwriter.writerow(headers)

        time_offset = 0
        time_offset_check = True

        for orientation in orientations:
            try:
                ser = serial.Serial(port, 115200, timeout=1)
            except serial.SerialException as e:
                status_queue.put(f"Failed to connect to {port}: {str(e)}")
                return

            # Wait for user to start
            count = 0
            while True:
                count += 1
                incoming_data = ser.readline().decode('utf-8', errors='ignore').strip()
                if count <= 1:
                    time.sleep(2)
                    status_queue.put(f"Press 'space' to start {orientation} for 5 seconds")
                if incoming_data == "#" or keyboard.is_pressed('space'):
                    status_queue.put("Starting")
                    break
            
            status_queue.put(f"Collecting data for {orientation}...")
            start_time = time.time()
            temp_data = []

            while time.time() - start_time < 3:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('$'):
                    status_queue.put(f"Reset detected at {float(line.split(',')[1])*1e-6}")
                    ser.close()
                    return
                data = line.split(',')
                if len(data) == 11 - 3 - 2:  # Match original data format
                    try:
                        time_sec = float(data[0]) * 1e-6
                        acx1 = float(data[3])
                        acy1 = float(data[4])
                        acz1 = float(data[5])
                        if time_offset_check:
                            time_offset = time_sec
                            time_offset_check = False
                        time_sec -= time_offset
                        csvwriter.writerow([time_sec, acx1, acy1, acz1, orientation])
                        temp_data.append([acx1, acy1, acz1])
                    except (ValueError, IndexError):
                        continue

            if temp_data:
                avg_sample = np.mean(temp_data, axis=0)
                stationary_samples.append(avg_sample)
                status_queue.put(f"Data collected for {orientation}")
                ser.close()
            else:
                status_queue.put(f"Warning: No data collected for {orientation}")
                ser.close()
                return

    ser.close()

    # Perform calibration
    if len(stationary_samples) < 6:
        status_queue.put("Error: Insufficient data for calibration (need 6 orientations)")
        return
    measurements = np.array(stationary_samples)
    offsets, gains = compute_accel_calibration(measurements, status_queue)

    # Save results
    parent_folder = r'AllInOne'
    os.makedirs(parent_folder, exist_ok=True)
    summary_path = os.path.join(parent_folder, 'accel_calibration_history.csv')
    with open(summary_path, 'w', newline='') as summaryfile:
        summarywriter = csv.writer(summaryfile)
        summarywriter.writerow(['Date', 'Offset X', 'Offset Y', 'Offset Z', 'Gain X', 'Gain Y', 'Gain Z'])
        summarywriter.writerow([datetime.now().strftime("%m_%d_%Y"), *offsets, *gains])

    result = (f"Offsets: x={offsets[0]:.6f}, y={offsets[1]:.6f}, z={offsets[2]:.6f}\n"
              f"Gains: x={gains[0]:.6f}, y={gains[1]:.6f}, z={gains[2]:.6f}")
    status_queue.put(f"Calibration completed:\n{result}")
    return result
