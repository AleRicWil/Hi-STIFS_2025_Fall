import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import os
from skopt import gp_minimize
from skopt.space import Real
from scipy.signal import savgol_filter

local_run_flag = False

class StalkInteraction:
    def __init__(self, time, force, position, section):
        self.start_time = time[0]
        self.end_time = time[-1]
        self.time_loc = np.average(time)
        self.time = time
        self.force = force
        self.position = position
        self.fits = {}
        self.height = section.height# * 0.885
        self.yaw = section.yaw# * 1.48

        # Force vs deflection
        alpha = np.radians(38.0)
        self.pos_x = (section.max_position - self.position)*np.sin(self.yaw)/np.cos(alpha)
        # self.force = self.force/np.cos(self.yaw - alpha)

    def filter_data(self, time, force, position, pos_x, force_D_pos_x):
        count = 0
        prev_len = len(time)
        self.time_filt = time
        self.force_filt = force
        self.position_filt = position
        self.pos_x_filt = pos_x
        self.force_D_pos_x_filt = force_D_pos_x
        
        while self.p90_FDX - self.p10_FDX > 700 or self.p10_FDX < 30 or max(self.force_D_pos_x_filt) - self.p90_FDX > 700:
            if len(self.time_filt) <= 100:
                break
            count += 1
            mask = (self.force_D_pos_x_filt < self.p90_FDX) & (self.force_D_pos_x_filt > self.p10_FDX)
            self.force_filt = self.force_filt[mask]
            self.position_filt = self.position_filt[mask]
            self.pos_x_filt = self.pos_x_filt[mask]
            self.force_D_pos_x_filt = self.force_D_pos_x_filt[mask]
            self.time_filt = self.time_filt[mask]
            
            self.slope_f, self.intercept_f, self.r_f, _, _ = stats.linregress(self.time_filt, self.force_filt)
            self.slope_p, self.intercept_p, self.r_p, _, _ = stats.linregress(self.time_filt, self.position_filt)
            self.slope_fx, self.intercept_fx, self.r_fx, _, _ = stats.linregress(self.pos_x_filt, self.force_filt)
            self.p10_FDX, self.p90_FDX = np.percentile(self.force_D_pos_x_filt, [10, 90])
            self.avg_FDX = np.mean(self.force_D_pos_x_filt)
            
        return count

    def plot_filtered_data(self, count):
        fit_f = np.polyval([self.slope_f, self.intercept_f], self.time_filt)
        fit_p = np.polyval([self.slope_p, self.intercept_p], self.time_filt)
        fit_fx = np.polyval([self.slope_fx, self.intercept_fx], self.pos_x_filt)
        
        print(f'Ending iterative fit with {len(self.time_filt)} points\nR^2: {self.r_f**2}, {self.r_p**2}')
        fig, ax = plt.subplots(2, 2, sharex=False, figsize=(10,10))
        ax[0,0].scatter(self.time_filt, self.force_filt, s=5)
        ax[0,0].plot(self.time_filt, fit_f, c='orange')
        ax[0,0].set_ylim(0, 60)
        ax[0,0].set_ylabel('Force (N)')

        ax[1,0].scatter(self.time_filt, self.position_filt, s=5)
        ax[1,0].plot(self.time_filt, fit_p, c='orange')
        ax[1,0].set_ylim(0, 0.20)
        ax[1,0].set_ylabel('Position (m)')
        ax[1,0].set_xlabel('Time (s)')

        ax[0,1].plot(self.pos_x, self.force, linewidth=0.5, c='red')
        ax[0,1].scatter(self.pos_x_filt, self.force_filt, s=5)
        ax[0,1].plot(self.pos_x_filt, fit_fx, c='orange')
        for group in self.force_clean_groups:
            pos_x = [point['x'] for point in group['points']]
            force = [point['y'] for point in group['points']]
            ax[0,1].scatter(pos_x, force)
        ax[0,1].set_xlim(0, 0.05)

        ax[1,1].plot(self.pos_x, self.force_D_pos_x, linewidth=0.5, c='red')
        ax[1,1].set_ylim(min(self.force_D_pos_x_filt)*0.9, max(self.force_D_pos_x_filt)*1.1)
        ax[1,1].set_xlim(0, 0.05)
        ax[1,1].scatter(self.pos_x_filt, self.force_D_pos_x_filt, s=5)
        ax[1,1].axhline(self.avg_FDX, linewidth=0.5, c='green')
        ax[1,1].axhline(self.p10_FDX, linewidth=0.5, c='blue')
        ax[1,1].axhline(self.p90_FDX, linewidth=0.5, c='red')
        plt.suptitle(f'{len(self.time_filt)} of {len(self.time)} points. Filtered {count} times\nSlope_FX: {self.slope_fx:.1f}, Avg_FDX: {self.avg_FDX:.1f}\nf/p: {-self.slope_f/self.slope_p/np.sin(self.yaw):.1f}, R_fx^2: {self.r_fx**2:.3f}')
        plt.show()

    def calc_stalk_stiffness(self):
        self.force_D_pos_x = np.gradient(self.force, self.pos_x)
        self.p10_FDX, self.p90_FDX = np.percentile(self.force_D_pos_x, [10, 90])
        self.avg_FDX = np.mean(self.force_D_pos_x)
        self.slope_fx, self.intercept_fx, self.r_fx, _, _ = stats.linregress(self.pos_x, self.force)
        self.slope_f, self.intercept_f, self.r_f, _, _ = stats.linregress(self.time, self.force)
        self.slope_p, self.intercept_p, self.r_p, _, _ = stats.linregress(self.time, self.position)
        
        count = self.filter_data(self.time, self.force, self.position, self.pos_x, self.force_D_pos_x)
        self.clean_FX_data()
        # self.plot_filtered_data(count)

        
        self.fits['time'] = self.time_filt
        self.fits['force'] = np.polyval([self.slope_f, self.intercept_f], self.time_filt)
        self.fits['position'] = np.polyval([self.slope_p, self.intercept_p], self.time_filt)
        self.time_loc = (self.time_filt[0] + self.time_filt[-1]) / 2
        self.stiffness = (self.slope_fx * self.height**3) / 3

    def clean_FX_data(self):
        # reassign self arrays for function use
        t = self.time_filt
        x = self.pos_x_filt
        y = self.force_filt
        s = self.position_filt
        dydx = self.force_D_pos_x_filt

        # Keep indices where x is strictly decreasing over reverse time
        keep_indices = [len(x) - 1]
        for i in range(len(x) - 2, -1, -1):
            if x[i] < x[keep_indices[-1]]:
                keep_indices.append(i)
        
        keep_indices.reverse()
        t = [t[i] for i in keep_indices]
        x = [x[i] for i in keep_indices]
        y = [y[i] for i in keep_indices]
        dydx = [dydx[i] for i in keep_indices]
        s = [s[i] for i in keep_indices]

        # Group points into increasing force segments
        y_min, y_max = np.min(y), np.max(y)
        y_span = y_max - y_min
        y_gaps = np.diff(y, append=0)
        y_groups = []
        y_vals = []; x_vals = []; t_vals = []; dydx_vals = []; s_vals = []
        current_segment = []
        threshold = y_span * 0.15

        for gap, x_val, y_val, t_val, dydx_val, s_val in zip(y_gaps, x, y, t, dydx, s):
            point = {'x': x_val, 'y': y_val, 't': t_val, 'dydx': dydx_val, 's': s_val}
            if gap < threshold:
                current_segment.append(point)
                x_vals.append(x_val); y_vals.append(y_val); t_vals.append(t_val); dydx_vals.append(dydx_val); s_vals.append(s_val)
            else:
                if current_segment:
                    current_segment.append(point)
                    x_vals.append(x_val); y_vals.append(y_val); t_vals.append(t_val); dydx_vals.append(dydx_val); s_vals.append(s_val)
                    y_groups.append({'points': current_segment, 'avg_force': np.mean(y_vals), 'avg_x': np.mean(x_vals), 
                                     'avg_t': np.mean(t_vals), 't_span': t_vals[-1] - t_vals[0], 'avg_dydx': np.mean(dydx_vals),  'num_points': len(y_vals)})
                current_segment = []
                y_vals = []
                x_vals = []
        if current_segment:
            current_segment.append(point)  # Include last point
            x_vals.append(x_val); y_vals.append(y_val); t_vals.append(t_val); dydx_vals.append(dydx_val); s_vals.append(s_val)
            y_groups.append({'points': current_segment, 'avg_force': np.mean(y_vals), 'avg_x': np.mean(x_vals), 
                             'avg_t': np.mean(t_vals),  't_span': t_vals[-1] - t_vals[0], 'avg_dydx': np.mean(dydx_vals), 'num_points': len(y_vals)})
        
        # remove bad initial sections where stalk weakly touches sensor
        if len(y_groups) == 2 and \
            y_groups[1]['avg_force'] - y_groups[0]['avg_force'] > y_span*0.6 and \
            y_groups[1]['num_points'] > y_groups[0]['num_points']*1.2 and \
            y_groups[1]['t_span'] > y_groups[0]['t_span']*2.0 and \
            y_groups[0]['t_span'] < 0.4 and \
            abs(y_groups[0]['avg_dydx'] - y_groups[1]['avg_dydx']) > 5:
            del y_groups[0]

        # Write all points in y_groups to t, x, y, dydx, s
        t = []; x = []; y = []; dydx = []; s = []
        for group in y_groups:
            for point in group['points']:
                t.append(point['t']); x.append(point['x']); y.append(point['y']); dydx.append(point['dydx']); s.append(point['s'])


        # Update self arrays
        self.time_clean = t
        self.pos_x_clean = x
        self.force_clean = y
        self.force_D_pos_x_clean = dydx
        self.position_clean = s
        self.force_clean_groups = y_groups

        self.slope_fx, self.intercept_fx, self.r_fx, _, _ = stats.linregress(self.pos_x_clean, self.force_clean)
        self.slope_f, self.intercept_f, self.r_f, _, _ = stats.linregress(self.time_clean, self.force_clean)
        self.slope_p, self.intercept_p, self.r_p, _, _ = stats.linregress(self.time_clean, self.position_clean)


class LabStalkRow:
    def __init__(self, date, test_num, min_force_rate=-0.5, pos_accel_tol=1.0, force_accel_tol=700):
        # These params set the filter bounds for identifying which portions of the data are stalk interactions. These are ideally straight lines, increasing in
        # force and decreasing in position. 
            # this window should be a bit wider than the physical sensor
        self.min_position = 5*1e-2  # centimeters, location of 2nd strain gauge
        self.max_position = 18*1e-2 # centimeters, location of beam end
        self.min_force = 0.5 # newton, easily cut out the spaces between stalks. Rejects noisy detection regime
        self.min_force_rate = min_force_rate    # newton/sec, only look at data where the stalk is being pushed outward
        self.max_force_rate = 70                # newton/sec, reject stalk first falling onto sensor beam
        self.max_pos_rate = 0.05                # m/sec, only look at data where stalk is moving forward (decreasing) on sensor beam, allow some jitter
        self.pos_accel_tol = pos_accel_tol  # m/s^2, reject curvature on either side of good stalk interaction 
        self.force_accel_tol = force_accel_tol # newton/s^2, reject curvature on either side of good stalk interaction
        self.min_seq_points = 10    # don't consider little portions of data that pass the initial filter
        self.stitch_gap_limit = 80  # if two good segments are close together, stitch into one segment (including the gap)

        # Results folder
        parent_folder = r'Results'
        os.makedirs(parent_folder, exist_ok=True)   # create the folder if it doesn't already exist
        self.results_path = os.path.join(parent_folder, r'results.csv')   # one file to store all results from all dates/sections
            # these are used to find the file, and written alongside the results in the results file
        self.date = date   
        self.test_num = test_num

        # Load data and stored calibration from data collection CSV header
        self.csv_path = rf'Raw Data\{date}\{date}_test_{test_num}.csv'
        if not os.path.exists(self.csv_path):
            self.exist = False
            return
        with open(self.csv_path, 'r') as f:
            self.exist = True
            reader = csv.reader(f)  # create an object which handles the CSV operations
            self.header_rows = []
            for row in reader:  # grab all the header rows
                if row[0] == '=====':   # this string (in the first column) divides the header from the data
                    break
                self.header_rows.append(row)
            
            params_read = 0 # track how many parameters have been read
            for row in self.header_rows:
                    # the first column of each row is the parameter name. The second column is the parameter's value
                if row[0] == "configuration":
                    self.configuration = row[1]
                    params_read += 1
                if row[0] == "sensor calibration (k d c)":
                        # c values are not used, instead calculated from initial idle values of each data collection
                        # read in the strings and convert to floats
                    k_str = row[1]
                    d_str = row[2]
                    k_values = [float(v) for v in k_str.split()]
                    d_values = [float(v) for v in d_str.split()]
                    self.k_1, self.k_2, self.k_B1, self.k_B2 = k_values
                    self.d_1, self.d_2, self.d_B1, self.d_B2 = d_values
                    params_read += 1
                if row[0] == "stalk array (lo med hi)":
                        # this is which block or section in the field (or synthetic stalk type in the lab)
                    self.stalk_type = row[1]
                    params_read += 1
                if row[0] == "sensor height (cm)":
                    self.height = float(row[1])*1e-2    # meters in this code, stored as cm in header
                    params_read += 1
                if row[0] == "sensor yaw (degrees)":
                    self.yaw = np.radians(float(row[1]))    # radians in this code
                    params_read += 1
                if row[0] == "sensor offset (cm to gauge 2)":
                    self.sensor_offset = float(row[1])*1e-2 # meters in this code, but it isn't used anywhere
                    params_read += 1
            if not params_read >= 6:
                raise ValueError("Test parameter rows missing in header")
            
            # Read the data
            data = pd.read_csv(f)   # load the CSV with pandas from where the reader object left off. Must be done within 'with open() as f:' scope

            # store each column by its title and covert the pandas object to a numpy array
        self.time = data['Time'].to_numpy()
        self.strain_1 = self.strain_1_raw = data['Strain A1'].to_numpy()    # CSV labels as A1/A2 for backwards compatability with 4 channel sensors. This code
        self.strain_2 = self.strain_2_raw = data['Strain A2'].to_numpy() 
        self.strain_B1 = self.strain_B1_raw = data['Strain B1'].to_numpy()
        self.strain_B2 = self.strain_B2_raw = data['Strain B2'].to_numpy()
        mask = (self.time >= 0.001) & (self.time <= 600) & (np.diff(self.time, prepend=self.time[0]) >= 0)
        self.time = self.time[mask]
        self.strain_1 = self.strain_1_raw = self.strain_1_raw[mask]
        self.strain_2 = self.strain_2_raw = self.strain_2_raw[mask]
        self.strain_B1 = self.strain_B1_raw = self.strain_B1_raw[mask]
        self.strain_B2 = self.strain_B2_raw = self.strain_B2_raw[mask]
            # check if this data collection had an accelerometer
        self.accel_flag = False  #
        if 'AcX1' in data.columns:
            self.accel_flag = True
            self.acX = self.acX_raw = data['AcX1'].to_numpy()   # CSV is set up to allow two accelerometers
            self.acY = self.acY_raw = data['AcY1'].to_numpy()
            self.acZ = self.acZ_raw = data['AcZ1'].to_numpy()
            self.acX = self.acX_raw = self.acX_raw[mask]
            self.acY = self.acY_raw = self.acY_raw[mask]
            self.acZ = self.acZ_raw = self.acZ_raw[mask]

    def smooth_raw_data(self, strain_window=20, strain_order=1, accel_window=50, accel_order=1):
        self.strain_1 = savgol_filter(self.strain_1, strain_window, strain_order)
        self.strain_2 = savgol_filter(self.strain_2, strain_window, strain_order)
        self.strain_B1 = savgol_filter(self.strain_B1, strain_window, strain_order)
        self.strain_B2 = savgol_filter(self.strain_B2, strain_window, strain_order)
        if self.accel_flag:
            self.acX = savgol_filter(self.acX, accel_window, accel_order)
            self.acY = savgol_filter(self.acY, accel_window, accel_order)
            self.acZ = savgol_filter(self.acZ, accel_window, accel_order)

    def correct_linear_drift(self, zero_cutoff=200):
        t_start = self.time[0]
        t_end = self.time[-1]

        self.strain_1_ini = np.mean(self.strain_1[0:zero_cutoff])
        self.strain_b1_ini = np.mean(self.strain_b1[0:zero_cutoff])
        self.strain_2_ini = np.mean(self.strain_2[0:zero_cutoff])
        self.strain_b2_ini = np.mean(self.strain_b2[0:zero_cutoff])

        strain_a1_end = np.mean(self.strain_1[-zero_cutoff:])
        strain_b1_end = np.mean(self.strain_b1[-zero_cutoff:])
        strain_a2_end = np.mean(self.strain_2[-zero_cutoff:])
        strain_b2_end = np.mean(self.strain_b2[-zero_cutoff:])

        a1_drift_slope = (strain_a1_end - self.strain_1_ini) / (t_end - t_start)
        b1_drift_slope = (strain_b1_end - self.strain_b1_ini) / (t_end - t_start)
        a2_drift_slope = (strain_a2_end - self.strain_2_ini) / (t_end - t_start)
        b2_drift_slope = (strain_b2_end - self.strain_b2_ini) / (t_end - t_start)

        self.a1_drift = a1_drift_slope * self.time + self.strain_1_ini
        self.b1_drift = b1_drift_slope * self.time + self.strain_b1_ini
        self.a2_drift = a2_drift_slope * self.time + self.strain_2_ini
        self.b2_drift = b2_drift_slope * self.time + self.strain_b2_ini

        corrected_a1 = self.strain_1 - (self.a1_drift - self.strain_1_ini)
        corrected_b1 = self.strain_b1 - (self.b1_drift - self.strain_b1_ini)
        corrected_a2 = self.strain_2 - (self.a2_drift - self.strain_2_ini)
        corrected_b2 = self.strain_b2 - (self.b2_drift - self.strain_b2_ini)

        self.strain_1 = corrected_a1
        self.strain_b1 = corrected_b1
        self.strain_2 = corrected_a2
        self.strain_b2 = corrected_b2

    def shift_initials(self, time_cutoff):
            cutoff_index = np.where(self.time - self.time[0] > time_cutoff)[0][0]
            self.strain_1_ini  = np.mean(self.strain_1[0:cutoff_index])
            self.strain_2_ini = np.mean(self.strain_2[0:cutoff_index])
            self.strain_B1_ini = np.mean(self.strain_B1[0:cutoff_index])
            self.strain_B2_ini = np.mean(self.strain_B2[0:cutoff_index]) 
            self.c_1 = self.strain_1_ini
            self.c_2 = self.strain_2_ini
            self.c_B1 = self.strain_B1_ini
            self.c_B2 = self.strain_B2_ini

    def calc_force_position(self, smooth=True, window=20, order=1, small_den_cutoff=2.5*1e-5):
        self.force_num = self.k_2*(self.strain_1 - self.c_1) - self.k_1*(self.strain_2 - self.c_2)
        self.force_den = self.k_1*self.k_2*(self.d_2 - self.d_1)
        self.force = self.force_raw = np.where(self.force_num / self.force_den < 0, 0, self.force_num / self.force_den)

        self.pos_num = self.k_2*self.d_2*(self.strain_1 - self.c_1) - self.k_1*self.d_1*(self.strain_2 - self.c_2)
        self.pos_den = self.k_2*(self.strain_1 - self.c_1) - self.k_1*(self.strain_2 - self.c_2)
        self.position = self.position_raw = np.clip(np.where(np.abs(self.pos_den) < small_den_cutoff, 0, self.pos_num/self.pos_den), 0, 0.30)

        self.force_numB = self.k_B2*(self.strain_B1 - self.c_B1) - self.k_B1*(self.strain_B2 - self.c_B2)
        self.force_denB = self.k_B1*self.k_B2*(self.d_B2 - self.d_B1)
        self.forceB = self.force_rawB = np.where(self.force_numB / self.force_denB < 0, 0, self.force_numB / self.force_denB)

        self.pos_numB = self.k_B2*self.d_B2*(self.strain_B1 - self.c_B1) - self.k_B1*self.d_B1*(self.strain_B2 - self.c_B2)
        self.pos_denB = self.k_B2*(self.strain_B1 - self.c_B1) - self.k_B1*(self.strain_B2 - self.c_B2)
        self.positionB = self.position_rawB = np.clip(np.where(np.abs(self.pos_denB) < small_den_cutoff, 0, self.pos_numB/self.pos_denB), 0, 0.30)

        if smooth:
            self.force = savgol_filter(self.force, window, order)
            self.position = savgol_filter(self.position, window, order)
            self.forceB = savgol_filter(self.forceB, window, order)
            self.positionB = savgol_filter(self.positionB, window, order)

    def differentiate_force_position(self, smooth=True, window=20, order=1):
        self.force_DT = np.gradient(self.force, self.time)
        self.position_DT = np.gradient(self.position, self.time)

        if smooth:
            self.force_DT = savgol_filter(self.force_DT, window, order)
            self.position_DT = savgol_filter(self.position_DT, window, order)

    def differentiate_force_position_DT(self, smooth=True, window=20, order=1):
        self.force_DDT = np.gradient(self.force_DT, self.time)
        self.position_DDT = np.gradient(self.position_DT, self.time)

        if smooth:
            self.force_DDT = savgol_filter(self.force_DDT, window, order)
            self.position_DDT = savgol_filter(self.position_DDT, window, order)

    def find_stalk_interaction(self):
        mask = (self.force > self.min_force) & \
            (self.position > self.min_position) & \
            (self.position < self.max_position) & \
            (self.force_DT > self.min_force_rate) & \
            (self.force_DT < self.max_force_rate) & \
            (self.position_DT < self.max_pos_rate)
        
        interaction_indices = np.where(mask)[0]
        if len(interaction_indices) == 0:
            self.interaction_indices = np.array([])
            return
        
        # Filter out blips (groups with fewer than min_sequential indices)
        if len(interaction_indices) > 0:
            diffs = np.diff(interaction_indices)
            group_starts = np.where(diffs > 1)[0] + 1
            groups = np.split(interaction_indices, group_starts)
            interaction_indices = np.concatenate([g for g in groups if len(g) >= self.min_seq_points]) if groups else np.array([])
        
        diffs = np.diff(interaction_indices)
        group_starts = np.where(diffs > 1)[0] + 1
        groups = np.split(interaction_indices, group_starts)
        filtered_groups = [g for g in groups if len(g) >= self.min_seq_points]
        
        stitched_indices = []
        prev_end = None
        for group in filtered_groups:
            if prev_end is not None:
                gap = group[0] - prev_end - 1
                if gap < self.stitch_gap_limit:
                    if abs(self.force[group[0]] - self.force[prev_end]) < 1 and \
                    abs(self.position[group[0]] - self.position[prev_end]) < 0.02:
                        stitched_indices.extend(range(prev_end + 1, group[0]))
            stitched_indices.extend(group)
            prev_end = group[-1]
        
        self.interaction_indices = np.array(stitched_indices)
        self.stalk_force = self.force[self.interaction_indices]
        self.stalk_position = self.position[self.interaction_indices]
        self.stalk_time = self.time[self.interaction_indices]

    def collect_stalks(self):
        if len(self.interaction_indices) == 0:
            print('No interactions')
            return
        
        gaps = np.diff(self.interaction_indices)
        split_points = np.where(gaps > self.stitch_gap_limit * 0.3)[0] + 1
        
        groups = np.split(self.interaction_indices, split_points)
        
        self.stalks = []
        
        for group in groups:
            if len(group) < self.min_seq_points:
                print('Not enough points')
                continue
            
            time = self.time[group]
            force = self.force[group]
            position = self.position[group]
            
            duration = time[-1] - time[0]
            if duration < 0.3:
                print('Not long enough')
                continue
            
            slope_f, intercept_f, r_f, _, _ = stats.linregress(time, force)
            slope_p, intercept_p, r_p, _, _ = stats.linregress(time, position)
            # print(slope_f, slope_p, r_f**2, r_p**2)
            
            if slope_f > 0 and slope_p < 0 and r_f**2 > 0.5 and r_p**2 > 0.5:
                stalk = StalkInteraction(time, force, position, self)
                self.stalks.append(stalk)
            
    def calc_stalk_stiffnesses(self):
        # print(f'Computing stiffness for {self.stalk_type} stalks')
        self.stiffnesses = []
        for stalk in self.stalks:
            if not np.isnan(stalk.time.all()):
                stalk.calc_stalk_stiffness()
                self.stiffnesses.append(stalk.stiffness)
            else:
                self.stiffnesses.append(np.nan)
            
    def plot_force_position(self, view_stalks=False, plain=True, show_accels=False):
        try:
            time_ini = self.stalks[0].time_loc
            time_end = self.stalks[-1].time_loc
        except:
            time_ini = 0
            time_end = 10
        if show_accels:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9.5, 4.8))
        else:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
        ax[0].plot(self.time - time_ini, self.force, label='Force')
        ax[0].plot(self.time - time_ini, self.force_raw, label='raw', linewidth=0.5)
        ax[0].plot(self.time - time_ini, self.forceB, label='Force_B')
        ax[0].set_ylabel('Force (N)')

        ax[1].plot(self.time - time_ini, self.position*100, label='Position')
        ax[1].plot(self.time - time_ini, self.positionB*100, label='Position_B')
        ax[1].plot(self.time - time_ini, self.position_raw*100, label='raw', linewidth=0.5)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (cm)')

        if show_accels:
            ax[2].plot(self.time - time_ini, self.pitch_smooth, label='Pitch')
            ax[2].plot(self.time - time_ini, self.roll_smooth, label='Roll')
            # ax[2].plot(self.time - time_ini, self.acZ, label='Vertical')
            ax[2].legend()

        plt.suptitle(f'{self.configuration}, Date:{self.date}, Test #{self.test_num}\nStalks:{self.stalk_type}')
        plt.xlim(-2, time_end - time_ini + 2)

        if view_stalks:
            for stalk in self.stalks:
                if not np.isnan(stalk.time).all():
                    ax[0].plot(stalk.time - time_ini, stalk.force, c='red')
                    ax[1].plot(stalk.time - time_ini, stalk.position*100, c='red')
                    if hasattr(stalk, 'fits'):
                        ax[0].plot(stalk.fits['time'] - time_ini, stalk.fits['force'], c='green')
                        ax[1].plot(stalk.fits['time'] - time_ini, stalk.fits['position']*100, c='green')
            plt.tight_layout()

            # fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
            # ax[0].plot(self.time, self.force_DT, label='Force Rate')
            # ax[0].scatter(self.time[self.interaction_indices], self.force_DT[self.interaction_indices], c='red', s=5)
            # ax[0].set_ylabel('Force Rate (N/s)')
            # ax[1].plot(self.time, self.position_DT*100, label='Position Rate')
            # ax[1].scatter(self.time[self.interaction_indices], self.position_DT[self.interaction_indices]*100, c='red', s=5)
            # ax[1].set_xlabel('Time (s)')
            # ax[1].set_ylabel('Position Rate (cm/s)')
            # ax[1].legend()
            # plt.tight_layout()

            # fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.8, 4.8))
            # ax[0].plot(self.time, self.force_DDT, label='Force Accel')
            # ax[0].scatter(self.time[self.interaction_indices], self.force_DDT[self.interaction_indices], c='red', s=5)
            # ax[0].set_ylabel('Force Accel (N/s2)')
            # ax[1].plot(self.time, self.position_DDT*100, label='Position Accel')
            # ax[1].scatter(self.time[self.interaction_indices], self.position_DDT[self.interaction_indices]*100, c='red', s=5)
            # ax[1].set_xlabel('Time (s)')
            # ax[1].set_ylabel('Position Accel (cm/s2)')
            # plt.tight_layout()
        
    def plot_force_position_DT(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.force_DT, label='Force Rate')
        ax[0].set_ylabel('Force Rate (N/s)')
        ax[1].plot(self.time, self.position_DT*100, label='Position Rate')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.position_DT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Rate (cm/s)')
        ax[1].legend()
        plt.tight_layout()

    def plot_force_position_DDT(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.force_DDT, label='Force Accel')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[0].plot(self.time[self.interaction_indices], self.force_DDT[self.interaction_indices], 'ro', markersize=2, label='Near-zero accel')
        ax[0].set_ylabel('Force Accel (N/s2)')
        ax[1].plot(self.time, self.positionDDT*100, label='Position Accel')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.position_DDT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Accel (cm/s2)')
        plt.tight_layout()

    def plot_raw_strain(self):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
        axs[0, 0].plot(self.time, self.strain_1_raw, linewidth=0.3)
        axs[0, 0].plot(self.time, self.strain_1, label='A1')
        axs[0, 0].axhline(self.strain_1_ini, c='red', linewidth=0.5)
        axs[0, 0].legend()
        axs[0, 1].plot(self.time, self.strain_2_raw, linewidth=0.3)
        axs[0, 1].plot(self.time, self.strain_2, label='A2')
        axs[0, 1].axhline(self.strain_2_ini, c='red', linewidth=0.5)
        axs[0, 1].legend()
        axs[1, 0].plot(self.time, self.strain_B1_raw, linewidth=0.3)
        axs[1, 0].plot(self.time, self.strain_B1, label='B1')
        axs[1, 0].axhline(self.strain_B1_ini, c='red', linewidth=0.5)
        axs[1, 0].legend()
        axs[1, 1].plot(self.time, self.strain_B2_raw, linewidth=0.3)
        axs[1, 1].plot(self.time, self.strain_B2, label='B2')
        axs[1, 1].axhline(self.strain_B2_ini, c='red', linewidth=0.5)
        axs[1, 1].legend()
        plt.tight_layout()

        # if self.accel_flag:
        #     max_x = np.max(self.acX)
        #     min_x = np.min(self.acX)
        #     mask_max_x = np.isclose(self.acX, max_x, atol=np.abs(max_x)*0.001)
        #     self.time_x_max = self.time[mask_max_x]
        #     self.acX1_max = self.acX[mask_max_x]
        #     mask_min_x = np.isclose(self.acX, min_x, atol=np.abs(min_x)*0.001)
        #     self.time_x_min = self.time[mask_min_x]
        #     self.acX1_min = self.acX[mask_min_x]
            
        #     max_y = np.max(self.acY)
        #     min_y = np.min(self.acY)
        #     mask_max_y = np.isclose(self.acY, max_y, atol=np.abs(max_y)*0.001)
        #     self.time_y_max = self.time[mask_max_y]
        #     self.acY1_max = self.acY[mask_max_y]
        #     mask_min_y = np.isclose(self.acY, min_y, atol=np.abs(min_y)*0.001)
        #     self.time_y_min = self.time[mask_min_y]
        #     self.acY1_min = self.acY[mask_min_y]

        #     max_z = np.max(self.acZ)
        #     min_z = np.min(self.acZ)
        #     mask_max_z = np.isclose(self.acZ, max_z, atol=np.abs(max_z)*0.001)
        #     self.time_z_max = self.time[mask_max_z]
        #     self.acZ1_max = self.acZ[mask_max_z]
        #     mask_min_z = np.isclose(self.acZ, min_z, atol=np.abs(min_z)*0.001)
        #     self.time_z_min = self.time[mask_min_z]
        #     self.acZ1_min = self.acZ[mask_min_z]

        #     fig2, axs2 = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(6, 8))
        #     axs2[0].plot(self.time, self.acX_raw, linewidth=0.3)
        #     axs2[0].plot(self.time, self.acX, label='acX')
        #     axs2[0].scatter(self.time_x_max, self.acX1_max, c='red')
        #     axs2[0].scatter(self.time_x_min, self.acX1_min, c='green')
        #     axs2[0].legend()
        #     axs2[1].plot(self.time, self.acY_raw, linewidth=0.3)
        #     axs2[1].plot(self.time, self.acY, label='acY')
        #     axs2[1].scatter(self.time_y_max, self.acY1_max, c='red')
        #     axs2[1].scatter(self.time_y_min, self.acY1_min, c='green')
        #     axs2[1].legend()
        #     axs2[2].plot(self.time, self.acZ_raw, linewidth=0.3)
        #     axs2[2].plot(self.time, self.acZ, label='acZ')
        #     axs2[2].scatter(self.time_z_max, self.acZ1_max, c='red')
        #     axs2[2].scatter(self.time_z_min, self.acZ1_min, c='green')
        #     axs2[2].legend()
        #     plt.tight_layout()

    def save_results(self, overwrite_result=False):
        # Prepare row data
        row = [self.date, self.csv_path]
        numeric_headers = ['sensor height (cm)', 'sensor yaw (degrees)', 'sensor pitch (degrees)', 'sensor roll (degrees)',
                           'rate of travel (ft/min)', 'angle of travel (degrees)', 'sensor offset (cm to gauge 2)']
        for header_row in self.header_rows:
            if header_row[0] == "sensor calibration (k d c)":
                row.append(header_row[1] + ' : ' + header_row[2] + ' : ' + header_row[3])  # Keep as string
            else:
                value = header_row[1] if len(header_row) > 1 else ''
                if header_row[0] in numeric_headers and value == '':
                    row.append(np.nan)  # Use np.nan for empty numeric fields
                elif header_row[0] in numeric_headers:
                    row.append(float(value))  # Cast numeric strings to float
                else:
                    row.append(value)  # Keep as string or empty string
        row += [float(self.pos_accel_tol), float(self.force_accel_tol)]
        for stalk in self.stiffnesses:
            row.append(float(stalk) if not np.isnan(stalk) else np.nan)

        # Check if file exists
        file_exists = os.path.isfile(self.results_path)
        headers = ['Date', 'File'] + [row[0] for row in self.header_rows if row[0] != "====="]
        headers += ['positionDDT_tol', 'forceDDT_tol']
        headers += ['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']

        # Define dtypes for columns
        dtype_dict = {
            'Date': 'object',
            'File': 'object',
            'user_note': 'object',
            'rodney configuration': 'object',
            'sensor calibration (k d c)': 'object',
            'stalk array (lo med hi)': 'object',
            'sensor height (cm)': 'float64',
            'sensor yaw (degrees)': 'float64',
            'sensor pitch (degrees)': 'float64',
            'sensor roll (degrees)': 'float64',
            'rate of travel (ft/min)': 'float64',
            'angle of travel (degrees)': 'float64',
            'sensor offset (cm to gauge 2)': 'float64',
            'positionDDT_tol': 'float64',
            'forceDDT_tol': 'float64',
            'Stalk1': 'float64',
            'Stalk2': 'float64',
            'Stalk3': 'float64',
            'Stalk4': 'float64',
            'Stalk5': 'float64',
            'Stalk6': 'float64',
            'Stalk7': 'float64',
            'Stalk8': 'float64',
            'Stalk9': 'float64'
        }

        if file_exists:
            df = pd.read_csv(self.results_path)
            # Ensure DataFrame has correct dtypes
            for col, dtype in dtype_dict.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            if 'File' in df.columns and self.csv_path in df['File'].values and overwrite_result:
                # Overwrite existing row
                for col, val in zip(headers, row):
                    df.loc[df['File'] == self.csv_path, col] = val
                df.to_csv(self.results_path, index=False)
                return

        # Write new row
        with open(self.results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(headers)
            csvwriter.writerow(row)

    def calc_angles(self):
        cal_csv_path = r'AllInOne\accel_calibration_history.csv'
        cal_data = pd.read_csv(cal_csv_path)
        latest_cal = cal_data.iloc[-1]

        self.m_x = latest_cal['Gain X']; self.b_x = latest_cal['Offset X']
        self.m_y = latest_cal['Gain Y']; self.b_y = latest_cal['Offset Y']
        self.m_z = latest_cal['Gain Z']; self.b_z = latest_cal['Offset Z']
        
        self.x_g = self.acX*self.m_x + self.b_x
        self.y_g = self.acY*self.m_y + self.b_y
        self.z_g = self.acZ*self.m_z + self.b_z

        # Calculate angles (in radians) about global x and y axes
        theta_x = np.arctan2(-self.y_g, np.sqrt(self.x_g**2 + self.z_g**2))  # Angle about global x-axis
        theta_y = np.arctan2(self.x_g, np.sqrt(self.y_g**2 + self.z_g**2))  # Angle about global y-axis

        self.pitch = np.degrees(theta_x)
        self.roll = np.degrees(theta_y)
        self.pitch_smooth = savgol_filter(self.pitch, 100, 2)
        self.roll_smooth = savgol_filter(self.roll, 100, 2)

    def plot_accels(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6))

        ax[0].plot(self.time, self.pitch_smooth, label='Pitch')
        ax[0].plot(self.time, self.roll_smooth, label='Roll')
        ax[1].plot(self.time, self.z_g, label='Vertical')
        ax[0].legend()

        ax[0].axhline(0, c='red', linewidth=0.3)
        ax[0].axhline(10, c='red', linewidth=0.3)
        ax[0].axhline(15.1, c='red', linewidth=0.3)
        ax[0].axhline(30.25, c='red', linewidth=0.3)
        ax[0].axhline(-10, c='red', linewidth=0.3)
        ax[0].axhline(-15.1, c='red', linewidth=0.3)
        ax[0].axhline(-30.25, c='red', linewidth=0.3)

        ax[1].axhline(1, c='red', linewidth=0.3)
        ax[1].axhline(0, c='red', linewidth=0.3)
        ax[1].axhline(-1, c='red', linewidth=0.3)

def boxplot_data(rodney_config, date=None, stalk_type=None, plot_num=20, offset=None, height=None):
    results_df = pd.read_csv(r'Results\results.csv')
    
    if not date == None:
        results_df = results_df[results_df['Date'] == date]
    if not stalk_type == None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    if not offset == None:
        results_df = results_df[results_df['sensor offset (cm to gauge 2)'] == offset]
    if not height == None:
        results_df = results_df[results_df['sensor height (cm)'] == height]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    offset = config_results['sensor offset (cm to gauge 2)'].iloc[0]
    height= config_results['sensor height (cm)'].iloc[0]
    print(offset, height)


    
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    med_stats = med_EIs.describe()
    med_mean = np.mean(med_stats.loc['mean'])

    plt.figure(plot_num)
    plt.title(fr'{rodney_config}, Offset: {offset}cm, Height: {height}cm, Avg: {med_mean:.2f} N/$m^2$')
    plt.ylabel('Flexural Stiffness (N/m^2)')
    for _, row in lo_EIs.iloc[2:].iterrows():
        plt.scatter(range(1, len(row) + 1), row, c='red', s=2)
    lo_EIs.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='none', color='red'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='red'),
                   medianprops=dict(color='black'))
    for _, row in med_EIs.iloc[2:].iterrows():
        plt.scatter(range(1, len(row) + 1), row, c='green', s=2)
    med_EIs.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='none', color='green'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='green'),
                   medianprops=dict(color='black'))
    plt.axhline(med_mean)
    for _, row in hi_EIs.iloc[2:].iterrows():
        plt.scatter(range(1, len(row) + 1), row, c='blue', s=2)
    hi_EIs.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='none', color='blue'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='blue'),
                   medianprops=dict(color='black'))
    plt.ylim(0, 50)

def get_stats(rodney_config, date=None, stalk_type=None, plot_num=None, offset=None, height=None):
    def get_ci(data, type_mean, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std)
        margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * std
        rel_margin = margin / type_mean if type_mean != 0 else 0  # Relative margin
        return {'interval': ci, 'mean': mean, 'margin': margin, 'rel_margin': rel_margin}
    
    results_df = pd.read_csv(r'Results\results.csv')
    if not date == None:
        results_df = results_df[results_df['Date'] == date]
    if not stalk_type == None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    if not offset == None:
        results_df = results_df[results_df['sensor offset (cm to gauge 2)'] == offset]
    if not height == None:
        results_df = results_df[results_df['sensor height (cm)'] == height]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    lo_stats = lo_EIs.describe()
    lo_mean = np.mean(lo_stats.loc['mean'])
    lo_ci = {col: get_ci(lo_EIs[col], lo_mean) for col in lo_EIs.columns} 
    lo_Margins = np.array([lo_ci[col]['margin'] for col in lo_EIs.columns])
    lo_relMargins = np.array([lo_ci[col]['rel_margin'] for col in lo_EIs.columns])

    med_stats = med_EIs.describe()
    med_mean = np.mean(med_stats.loc['mean'])
    med_ci = {col: get_ci(med_EIs[col], med_mean) for col in med_EIs.columns}
    med_Margins = np.array([med_ci[col]['margin'] for col in med_EIs.columns])
    med_relMargins = np.array([med_ci[col]['rel_margin'] for col in med_EIs.columns])

    hi_stats = hi_EIs.describe()
    hi_mean = np.mean(hi_stats.loc['mean'])
    hi_ci = {col: get_ci(hi_EIs[col], hi_mean) for col in hi_EIs.columns}
    hi_Margins = np.array([hi_ci[col]['margin'] for col in hi_EIs.columns])
    hi_relMargins = np.array([hi_ci[col]['rel_margin'] for col in hi_EIs.columns])

    all_Margins = np.append(np.append(lo_Margins, med_Margins), hi_Margins)
    all_Margins_mean = np.nanmean(all_Margins)
    all_Margins_median = np.nanmedian(all_Margins)

    all_relMargins = np.append(np.append(lo_relMargins, med_relMargins), hi_relMargins)
    all_relMargins_mean = np.nanmean(all_relMargins)
    all_relMargins_median = np.nanmedian(all_relMargins)

    medhi_relMargins = np.append(med_relMargins, hi_relMargins)
    medhi_relMargins_mean = np.nanmean(medhi_relMargins)
    medhi_relMargins_median = np.nanmedian(medhi_relMargins)
    # print(all_relMargins_median)

    if plot_num is not None:
        plt.figure(plot_num)
        plt.scatter(range(1, len(lo_relMargins)+1), lo_relMargins*100, label='lo', c='red')
        plt.scatter(range(1, len(med_relMargins)+1), med_relMargins*100, label='med', c='green')
        plt.scatter(range(1, len(hi_relMargins)+1), hi_relMargins*100, label='hi', c='blue')
        plt.axhline(all_relMargins_median*100, c='black', label='Median')
        plt.axhline(medhi_relMargins_median*100, c='brown', label='M/H Median')
        plt.ylim(0, 80)
        plt.ylabel('% Relative Error Margin')
        plt.title(rodney_config + f'\nRelative err: {all_relMargins_median*100:.1f}%, Absolute err: {all_Margins_median:.2f} N/m^2')
        plt.legend()

    return all_relMargins_mean, all_relMargins_median

# Modified process_data to use optimized parameters
def process_data(date, test_num, view=False, overwrite=False):
    test = LabStalkRow(date=date, test_num=test_num)
    if test.exist:
        test.smooth_raw_data()
        test.shift_initials(time_cutoff=1.0)
        test.calc_force_position()
        test.differentiate_force_position()
        test.differentiate_force_position_DT()
        test.find_stalk_interaction()
        test.collect_stalks()
        test.calc_stalk_stiffnesses()

    if view:
        test.plot_force_position(view_stalks=True)
        plt.show()
        if overwrite:
            test.save_results(overwrite_result=True)
        
        # test.plot_raw_strain()
        # test.plot_force_position_DT()
        # test.plot_force_position_DDT()
    else:
        test.save_results(overwrite_result=True)
    
    if not local_run_flag:
        plt.show()

def show_force_position(dates, test_nums, show_accels=False):
    for date in dates:
        for test_num in test_nums:
            test = LabStalkRow(date=date, test_num=test_num)
            if test.exist:
                test.smooth_raw_data()
                test.shift_initials(time_cutoff=1.0)
                test.calc_force_position()
                test.differentiate_force_position()
                test.differentiate_force_position_DT()
                # test.find_stalk_interaction()
                # test.collect_stalks()
                # test.calc_stalk_stiffnesses()
                # test.calc_angles()

                test.plot_raw_strain()
                test.plot_force_position(view_stalks=False, show_accels=show_accels)
    plt.show()

def correlation(rodney_config, date=None, stalk_type=None):
    from scipy.stats import linregress
    results_df = pd.read_csv(r'Results\results.csv')
    if not date == None:
        results_df = results_df[results_df['Date'] == date]
    if not stalk_type == None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    darling_results = pd.read_csv(r'Results\PVC Low Stiffness.csv')
    d_lo_EIs = darling_results[['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']]
    darling_results = pd.read_csv(r'Results\PVC Medium Stiffness.csv')
    d_med_EIs = darling_results[['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']]
    darling_results = pd.read_csv(r'Results\PVC High Stiffness.csv')
    d_hi_EIs = darling_results[['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']]

    # Rename Rodney columns to match Darling
    lo_EIs.columns = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    med_EIs.columns = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    hi_EIs.columns = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']

    # Combine data by stalk type
    datasets = [
        ('Low Stiffness', lo_EIs, d_lo_EIs),
        ('Medium Stiffness', med_EIs, d_med_EIs),
        ('High Stiffness', hi_EIs, d_hi_EIs)
    ]

    # Collect medians
    # print(lo_EIs, med_EIs, hi_EIs)
    rodney_medians = []
    darling_medians = []
    for _, rodney_data, darling_data in datasets:
        for stalk in rodney_data.columns:
            rodney_vals = rodney_data[stalk].dropna()
            darling_vals = darling_data[stalk].dropna()
            rodney_medians.append(np.median(rodney_vals) if len(rodney_vals) > 0 else np.nan)
            darling_medians.append(np.median(darling_vals) if len(darling_vals) > 0 else np.nan)

    rodney_medians = np.array(rodney_medians)
    darling_medians = np.array(darling_medians)
    slope, inter, r, _, _ = linregress(darling_medians, rodney_medians)
    
    plt.plot(darling_medians, slope*darling_medians + inter, c='black', linewidth=0.5)
    plt.scatter(darling_medians, rodney_medians, label=fr'Median $R^2$: {r**2:.4f} Slope: {slope:.3f}, Int: {inter:.2f}')

    plt.plot(darling_medians, darling_medians, c='blue', linewidth='0.5')
    plt.xlabel(r'Darling Stiffness (N/$m^2$)', fontsize=14)
    plt.ylabel(r'Hi-STIFS Stiffness (N/$m^2$)', fontsize=14)
    plt.axis('equal')
    # plt.title(f'Date: {date}\n'+ rf'Median $R^2$: {r**2:.4f}, Slope: {slope:.3f}')
    plt.title(rf'Median $R^2$: {r**2:.4f}', fontsize=18)
    # plt.show()

    repeated_d_medians = []
    all_r_values = []
    for name, rodney_data, darling_data in datasets:
        for stalk in rodney_data.columns:
            rodney_vals = rodney_data[stalk].dropna().tolist()
            darling_vals = darling_data[stalk].dropna()
            if len(rodney_vals) > 0 and len(darling_vals) > 0:
                d_median = np.median(darling_vals)
                repeated_d_medians.extend([d_median] * len(rodney_vals))
                all_r_values.extend(rodney_vals)

    slope, inter, r, _, _ = linregress(repeated_d_medians, all_r_values)
    plt.figure()
    plt.plot(darling_medians, slope*darling_medians + inter, c='black', linewidth=0.5)
    plt.scatter(repeated_d_medians, all_r_values, label=fr'Median $R^2$: {r**2:.4f} Slope: {slope:.3f}, Int: {inter:.2f}')

    plt.plot(darling_medians, darling_medians, c='blue', linewidth='0.5')
    plt.xlabel(r'Darling Stiffness (N/$m^2$)', fontsize=14)
    plt.ylabel(r'Hi-STIFS Stiffness (N/$m^2$)', fontsize=14)
    plt.axis('equal')
    # plt.title(f'Date: {date}\n'+ rf'All $R^2$: {r**2:.4f}, Slope: {slope:.3f}')
    plt.title(rf'All $R^2$: {r**2:.4f}', fontsize=18)
    plt.show()


if __name__ == "__main__":
    local_run_flag = True
    
    '''Batch run of same configuration'''
    # for i in range(1, 15+1):
    #     process_data(date='08_13', test_num=f'{i}', view=True, overwrite=True)
    show_force_position(dates=['09_29'], test_nums=[1,2,3,4,5,6,7,8,9,10])

    # boxplot_data(rodney_config='Integrated Beam Prototype 1', date='07_03', plot_num=104)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', date='07_10', plot_num=105)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', date='07_14', plot_num=106)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', date='07_10', plot_num=107)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', date='07_11', plot_num=108)
    # boxplot_data(rodney_config='Integrated Beam Printed Guide 1', date='07_16', plot_num=109)
    # boxplot_data(rodney_config='Integrated Beam Fillet 1', date='08_13', plot_num=110)

    # boxplot_data(rodney_config='Integrated Beam Fillet 1', date='08_21', plot_num=108, offset=6.4, height=80.3)
    # boxplot_data(rodney_config='Integrated Beam Fillet 1 Normal', date='08_21', plot_num=109)
    '''end batch run'''

    '''Statistics'''
    # print('1 mean, median', get_stats(rodney_config='Integrated Beam Prototype 1', plot_num=204))
    # print('2 mean, median', get_stats(rodney_config='Integrated Beam Prototype 2', date='07_10', plot_num=205))
    # print('2 mean, median', get_stats(rodney_config='Integrated Beam Prototype 2', date='07_14', plot_num=206))
    # print('3 mean, median', get_stats(rodney_config='Integrated Beam Prototype 3', date='07_10', plot_num=207))
    # print('3 mean, median', get_stats(rodney_config='Integrated Beam Prototype 3', date='07_11', plot_num=208))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Printed Guide 1', date='07_16', plot_num=209))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Fillet 1', date='07_24', plot_num=210))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Fillet 1', date='08_21', plot_num=211, offset=6.4, height=80.3))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Fillet 1 Normal', date='08_21', plot_num=212))
    '''end statistics'''

    '''Single file run and view full file. Does not save result'''
    # process_data(date='07_14', test_num='1', view=True)
    '''end single file run'''

    # Optimize parameters for a specific configuration
    # optimize_parameters(dates=['07_16'], rodney_config='Integrated Beam Printed Guide 1')

    # correlation(rodney_config='Integrated Beam Fillet 1', date='08_13')

    plt.show()