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
results = []

class StalkInteraction:
    def __init__(self, time, force, position, section):
        self.start_time = time[0]
        self.end_time = time[-1]
        self.time_loc = np.average(time)
        self.time = time
        self.force = force
        self.position = position
        self.fits = {}
        # self.stiffness = {}
        self.height = section.height# * 0.885
        self.yaw = section.yaw# * 1.48

        # Force vs deflection
        self.pos_x = (section.max_position - self.position)*np.sin(self.yaw)

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
    def __init__(self, date, test_num, min_force_rate=0.01,
                 lo_pos_accel_tol=1.0, med_pos_accel_tol=0.5, hi_pos_accel_tol=0.5,
                 lo_force_accel_tol=25.0, med_force_accel_tol=30.0, hi_force_accel_tol=40.0):
        
        self.min_position = 5 * 1e-2    # centimeters
        self.max_position = 18 * 1e-2   # centimeters
        self.min_force = 2  # Newtons
        self.min_force_rate = min_force_rate  # Newton/second
        self.min_sequential = 10
        # Create results folder
        parent_folder = os.path.join('Results')
        os.makedirs(parent_folder, exist_ok=True)
        self.results_path = os.path.join(parent_folder, f'results.csv')
        self.date = date
        self.test_num = test_num

        # Load strain data and calibration coefficients from CSV header
        self.csv_path = rf'Raw Data\{date}\{date}_test_{test_num}.csv'
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            self.header_rows = []
            for row in reader:
                if row[0] == "=====":
                    break
                self.header_rows.append(row)
            # Find test parameter rows
            param_rows = 0
            for row in self.header_rows:
                if row[0] == "rodney configuration":
                    self.configuration = row[1]
                    param_rows += 1
                if row[0] == "sensor calibration (k d c)":
                    k_str = row[1]
                    d_str = row[2]
                    k_values = [float(v) for v in k_str.split()]
                    d_values = [float(v) for v in d_str.split()]
                    self.k_A1, self.k_B1, self.k_A2, self.k_B2 = k_values
                    self.d_A1, self.d_B1, self.d_A2, self.d_B2 = d_values
                    param_rows += 1
                if row[0] == "stalk array (lo med hi)":
                    self.stalk_type = row[1]
                    if row[1] == "lo":
                        self.result_color = 'red'
                        self.pos_accel_tol = lo_pos_accel_tol
                        self.force_accel_tol = lo_force_accel_tol
                    elif row[1] == "med":
                        self.result_color = 'green'
                        self.pos_accel_tol = med_pos_accel_tol
                        self.force_accel_tol = med_force_accel_tol
                    elif row[1] == "hi":
                        self.result_color = 'blue'
                        self.pos_accel_tol = hi_pos_accel_tol
                        self.force_accel_tol = hi_force_accel_tol
                    param_rows += 1
                if row[0] == "sensor height (cm)":
                    self.height = float(row[1])*1e-2
                    param_rows += 1
                if row[0] == "sensor yaw (degrees)":
                    self.yaw = np.radians(float(row[1]))
                    param_rows += 1
                if row[0] == "sensor offset (cm to gauge 2)":
                    self.sensor_offset = float(row[1])
                    param_rows += 1
            if not param_rows >= 6:
                raise ValueError("Test parameter rows missing in header")
            # Read the data
            data = pd.read_csv(f)

        self.time = data['Time'].to_numpy()
        self.strain_1 = self.strain_1_raw = data['Strain A1'].to_numpy()
        self.strain_2 = self.strain_2_raw = data['Strain A2'].to_numpy()
        self.strain_b1 = self.strain_b1_raw = data['Strain B1'].to_numpy()
        self.strain_b2 = self.strain_b2_raw = data['Strain B2'].to_numpy()
        self.accel_flag = False
        if 'AcX1' in data.columns:
            self.accel_flag = True
            self.acX = self.acX_raw = data['AcX1'].to_numpy()
            self.acY = self.acY_raw = data['AcY1'].to_numpy()
            self.acZ = self.acZ_raw = data['AcZ1'].to_numpy()

    def smooth_strains(self, window=50, order=1):
        self.strain_1 = savgol_filter(self.strain_1, window, order)
        self.strain_2 = savgol_filter(self.strain_2, window, order)
        self.strain_b1 = savgol_filter(self.strain_b1, window, order) 
        self.strain_b2 = savgol_filter(self.strain_b2, window, order)

    def smooth_accels(self, window=10, order=1):
        self.acX = savgol_filter(self.acX, window, order)
        self.acY = savgol_filter(self.acY, window, order)
        self.acZ = savgol_filter(self.acZ, window, order)

    def smooth_strain_DTs(self, window=50, order=1):
        self.strainDT_1 = savgol_filter(self.strainDT_1, window, order)
        self.strainDT_2 = savgol_filter(self.strainDT_2, window, order)
        self.strainDT_b1 = savgol_filter(self.strainDT_b1, window, order)
        self.strainDT_b2 = savgol_filter(self.strainDT_b2, window, order)

        if self.accel_flag:
            self.acX_DT = savgol_filter(self.acX_DT, window, order)
            self.acY_DT = savgol_filter(self.acY_DT, window, order)
            self.acZ_DT = savgol_filter(self.acZ_DT, window, order)

    def differentiate_strains(self):
        self.strainDT_1 = np.gradient(self.strain_1, self.time)
        self.strainDT_b1 = np.gradient(self.strain_b1, self.time)
        self.strainDT_2 = np.gradient(self.strain_2, self.time)
        self.strainDT_b2 = np.gradient(self.strain_b2, self.time)

        if self.accel_flag:
            self.acX_DT = np.gradient(self.acX, self.time)
            self.acY_DT = np.gradient(self.acY, self.time)
            self.acZ_DT = np.gradient(self.acZ, self.time)

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

    def shift_initials(self, initial_force=0, initial_position=0, zero_cutoff=200):
        '''Adjusts the calibration coefficients so the initial force from
        calculate_force_position() equals initial_force'''
        self.strain_1_ini = np.mean(self.strain_1[0:zero_cutoff])
        self.strain_2_ini = np.mean(self.strain_2[0:zero_cutoff])
        self.strain_b1_ini = np.mean(self.strain_b1[0:zero_cutoff])
        self.strain_b2_ini = np.mean(self.strain_b2[0:zero_cutoff])

        self.c_1 = self.strain_1_ini
        self.c_2 = self.strain_2_ini
        self.c_B1 = self.strain_b1_ini
        self.c_B2 = self.strain_b2_ini

    def calc_force_position(self, smooth=True, window=100, order=2, small_den_cutoff=0.000035):
        self.force_num = self.k_A2 * (self.strain_1 - self.c_1) - self.k_A1 * (self.strain_2 - self.c_2)
        self.force_den = self.k_A1 * self.k_A2 * (self.d_A2 - self.d_A1)
        self.force = self.force_num / self.force_den
        
        self.pos_num = self.k_A2 * self.d_A2 * (self.strain_1 - self.c_1) - self.k_A1 * self.d_A1 * (self.strain_2 - self.c_2)
        self.pos_den = self.k_A2 * (self.strain_1 - self.c_1) - self.k_A1 * (self.strain_2 - self.c_2)
        self.position = np.where(np.abs(self.pos_den) < small_den_cutoff, 0, self.pos_num / self.pos_den)
        
        if smooth:
            self.raw_force = self.force
            self.raw_position = self.position
            self.force = savgol_filter(self.force, window, order)
            self.position = savgol_filter(self.position, window, order)

    def differentiate_force_position(self, smooth=True, window=100, order=2):
        self.forceDT = np.gradient(self.force, self.time)
        self.positionDT = np.gradient(self.position, self.time)

        if smooth:
            self.forceDT = savgol_filter(self.forceDT, window, order)
            self.positionDT = savgol_filter(self.positionDT, window, order)

    def differentiate_force_position_DT(self, smooth=True, window=100, order=2):
        self.forceDDT = np.gradient(self.forceDT, self.time)
        self.positionDDT = np.gradient(self.positionDT, self.time)

        if smooth:
            self.forceDDT = savgol_filter(self.forceDDT, window, order)
            self.positionDDT = savgol_filter(self.positionDDT, window, order)

    def find_stalk_interaction(self):
        # Combine conditions into a single boolean mask
        mask = (np.abs(self.positionDDT) < self.pos_accel_tol) & \
            (self.position > self.min_position) & \
            (self.position < self.max_position) & \
            (self.positionDT < 0.1) & \
            (self.force > self.min_force) & \
            (self.forceDT > self.min_force_rate) & \
            (np.abs(self.forceDDT) < self.force_accel_tol)
        
        # Get initial interaction indices
        interaction_indices = np.where(mask)[0]
        
        # Filter out blips (groups with fewer than min_sequential indices)
        if len(interaction_indices) > 0:
            diffs = np.diff(interaction_indices)
            group_starts = np.where(diffs > 1)[0] + 1
            groups = np.split(interaction_indices, group_starts)
            interaction_indices = np.concatenate([g for g in groups if len(g) >= self.min_sequential]) if groups else np.array([])
        
        # Reconnect gaps < 30% of average gap
        if len(interaction_indices) > 1:
            gaps = np.diff(interaction_indices)
            valid_gaps = gaps[gaps > 1]
            if len(valid_gaps) > 0:
                threshold = 0.3 * np.mean(valid_gaps)
                new_indices = []
                for i in range(len(interaction_indices) - 1):
                    new_indices.append(interaction_indices[i])
                    gap = interaction_indices[i + 1] - interaction_indices[i]
                    if gap > 1 and gap < threshold:
                        new_indices.extend(range(interaction_indices[i] + 1, interaction_indices[i + 1]))
                new_indices.append(interaction_indices[-1])
                interaction_indices = np.array(new_indices, dtype=np.int64)
        
        # Assign results
        self.interaction_indices = interaction_indices
        self.stalk_force = self.force[interaction_indices]
        self.stalk_position = self.position[interaction_indices]
        self.stalk_time = self.time[interaction_indices]

    def collect_stalk_sections(self):
        gaps = np.concatenate(([0], np.diff(self.interaction_indices)))
        big_gaps = gaps[gaps>1]
        avg_gap = np.average(big_gaps)

        self.stalk_forces = []
        self.stalk_positions = []
        self.stalk_times = []
        force = []
        position = []
        time = []
        self.stalks = []
        
        for i in range(len(self.interaction_indices)):
            if gaps[i] <= 1:    # accumulate point on current stalk
                force.append(self.stalk_force[i])
                position.append(self.stalk_position[i])
                time.append(self.stalk_time[i])
            else:               # store current stalk and reset accumulation
                self.stalk_forces.append(np.array(force))
                self.stalk_positions.append(np.array(position))
                self.stalk_times.append(np.array(time))
                stalk = StalkInteraction(np.array(time), np.array(force), np.array(position), self)
                self.stalks.append(stalk)
                force = []
                position = []
                time = []

                
                if gaps[i] >= avg_gap*1.6:  # if the gap is very large, skip next stalk number
                    self.stalk_forces.append(np.nan)
                    self.stalk_positions.append(np.nan)
                    self.stalk_times.append(np.nan)
        
        # add the last stalk
        self.stalk_forces.append(np.array(force))
        self.stalk_positions.append(np.array(position))
        self.stalk_times.append(np.array(time))
        stalk = StalkInteraction(np.array(time), np.array(force), np.array(position), self)
        self.stalks.append(stalk)
            
    def calc_stalk_stiffnesses(self):
        # print(f'Computing stiffness for {self.stalk_type} stalks')
        self.force_fits = []
        self.position_fits = []
        self.flex_stiffs = []
        self.alt_flex_stiffs = []
        for i in range(len(self.stalk_times)):
            if not np.isnan(self.stalk_times[i]).all():
                time = self.stalk_times[i]
                force = self.stalk_forces[i]
                position = self.stalk_positions[i]
                
                # Fit linear regression to force vs time
                force_coeffs = np.polyfit(time, force, 1)
                force_slope = force_coeffs[0]  # Slope of force over time
                force_fit = np.polyval(force_coeffs, time)
                self.force_fits.append(force_fit)
                
                # Fit linear regression to position vs time
                pos_coeffs = np.polyfit(time, position, 1)
                pos_slope = pos_coeffs[0]  # Slope of position over time
                pos_fit = np.polyval(pos_coeffs, time)
                self.position_fits.append(pos_fit)

                # Calulate fluxual stiffness from slopes and system parameters
                num = force_slope*self.height**3
                den = -3*pos_slope*np.sin(self.yaw) # negate because position starts at end of sensor beam and ends at base
                flexural_stiffness = num/den
                self.flex_stiffs.append(flexural_stiffness)
            else:
                self.force_fits.append(np.nan)
                self.position_fits.append(np.nan)
                self.flex_stiffs.append(np.nan)

        for stalk in self.stalks:
            if not np.isnan(stalk.time.all()):
                stalk.calc_stalk_stiffness()
                self.alt_flex_stiffs.append(stalk.stiffness)
            else:
                self.alt_flex_stiffs.append(np.nan)
        self.flex_stiffs = self.alt_flex_stiffs
            
        results.append(self.alt_flex_stiffs)
            
    def plot_force_position(self, view_stalks=False, plain=True):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
        ax[0].plot(self.time, self.force, label='Force')
        ax[0].set_ylabel('Force (N)')
        ax[1].plot(self.time, self.position*100, label='Position')
        if hasattr(self, 'raw_force'):
            ax[0].plot(self.time, self.raw_force, label='raw', linewidth=0.5)
            ax[1].plot(self.time, self.raw_position*100, label='raw', linewidth=0.5)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (cm)')

        plt.suptitle(f'{self.configuration}, Date:{self.date}, Test #{self.test_num}\nStalks:{self.stalk_type}')

        
        if view_stalks:
            for i in range(len(self.stalk_times)):
                if not np.isnan(self.stalk_times[i]).all():
                    ax[0].plot(self.stalk_times[i], self.stalk_forces[i], c='red')
                    ax[0].plot(self.stalk_times[i], self.force_fits[i], c='green')
                    ax[1].plot(self.stalk_times[i], self.stalk_positions[i]*100, c='red')
                    ax[1].plot(self.stalk_times[i], self.position_fits[i]*100, c='green')
            plt.tight_layout()
        # For verifying calibration coefficients
        elif not plain:
            g = 9.8
            ax[0].axhline(1*g, c='red', linewidth=0.5)
            ax[0].axhline(0.5*g, c='red', linewidth=0.5)
            ax[0].axhline(0.2*g, c='red', linewidth=0.5)
            ax[1].axhline(15, c='red', linewidth=0.5)
            ax[1].axhline(10, c='red', linewidth=0.5)

    def plot_force_position_DT(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.forceDT, label='Force Rate')
        ax[0].set_ylabel('Force Rate (N/s)')
        ax[1].plot(self.time, self.positionDT*100, label='Position Rate')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.positionDT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Rate (cm/s)')
        ax[1].legend()
        plt.tight_layout()

    def plot_force_position_DDT(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.forceDDT, label='Force Accel')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[0].plot(self.time[self.interaction_indices], self.forceDDT[self.interaction_indices], 'ro', markersize=2, label='Near-zero accel')
        ax[0].set_ylabel('Force Accel (N/s2)')
        ax[1].plot(self.time, self.positionDDT*100, label='Position Accel')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.positionDDT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Accel (cm/s2)')
        plt.tight_layout()

    def plot_raw_strain(self):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
        axs[0, 0].plot(self.time, self.strain_1_raw, linewidth=0.3)
        axs[0, 0].plot(self.time, self.a1_drift, linewidth=0.3)
        axs[0, 0].plot(self.time, self.strain_1, label='A1')
        axs[0, 0].axhline(self.strain_1_ini, c='red', linewidth=0.5)
        axs[0, 0].legend()
        axs[0, 1].plot(self.time, self.strain_2_raw, linewidth=0.3)
        axs[0, 1].plot(self.time, self.a2_drift, linewidth=0.3)
        axs[0, 1].plot(self.time, self.strain_2, label='A2')
        axs[0, 1].axhline(self.strain_2_ini, c='red', linewidth=0.5)
        axs[0, 1].legend()
        axs[1, 0].plot(self.time, self.strain_b1_raw, linewidth=0.3)
        axs[1, 0].plot(self.time, self.b1_drift, linewidth=0.3)
        axs[1, 0].plot(self.time, self.strain_b1, label='B1')
        axs[1, 0].axhline(self.strain_b1_ini, c='red', linewidth=0.5)
        axs[1, 0].legend()
        axs[1, 1].plot(self.time, self.strain_b2_raw, linewidth=0.3)
        axs[1, 1].plot(self.time, self.b2_drift, linewidth=0.3)
        axs[1, 1].plot(self.time, self.strain_b2, label='B2')
        axs[1, 1].axhline(self.strain_b2_ini, c='red', linewidth=0.5)
        axs[1, 1].legend()
        plt.tight_layout()

        if self.accel_flag:
            max_x = np.max(self.acX)
            min_x = np.min(self.acX)
            mask_max_x = np.isclose(self.acX, max_x, atol=np.abs(max_x)*0.001)
            self.time_x_max = self.time[mask_max_x]
            self.acX1_max = self.acX[mask_max_x]
            mask_min_x = np.isclose(self.acX, min_x, atol=np.abs(min_x)*0.001)
            self.time_x_min = self.time[mask_min_x]
            self.acX1_min = self.acX[mask_min_x]
            
            max_y = np.max(self.acY)
            min_y = np.min(self.acY)
            mask_max_y = np.isclose(self.acY, max_y, atol=np.abs(max_y)*0.001)
            self.time_y_max = self.time[mask_max_y]
            self.acY1_max = self.acY[mask_max_y]
            mask_min_y = np.isclose(self.acY, min_y, atol=np.abs(min_y)*0.001)
            self.time_y_min = self.time[mask_min_y]
            self.acY1_min = self.acY[mask_min_y]

            max_z = np.max(self.acZ)
            min_z = np.min(self.acZ)
            mask_max_z = np.isclose(self.acZ, max_z, atol=np.abs(max_z)*0.001)
            self.time_z_max = self.time[mask_max_z]
            self.acZ1_max = self.acZ[mask_max_z]
            mask_min_z = np.isclose(self.acZ, min_z, atol=np.abs(min_z)*0.001)
            self.time_z_min = self.time[mask_min_z]
            self.acZ1_min = self.acZ[mask_min_z]

            fig2, axs2 = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(6, 8))
            axs2[0].plot(self.time, self.acX_raw, linewidth=0.3)
            axs2[0].plot(self.time, self.acX, label='acX')
            axs2[0].scatter(self.time_x_max, self.acX1_max, c='red')
            axs2[0].scatter(self.time_x_min, self.acX1_min, c='green')
            axs2[0].legend()
            axs2[1].plot(self.time, self.acY_raw, linewidth=0.3)
            axs2[1].plot(self.time, self.acY, label='acY')
            axs2[1].scatter(self.time_y_max, self.acY1_max, c='red')
            axs2[1].scatter(self.time_y_min, self.acY1_min, c='green')
            axs2[1].legend()
            axs2[2].plot(self.time, self.acZ_raw, linewidth=0.3)
            axs2[2].plot(self.time, self.acZ, label='acZ')
            axs2[2].scatter(self.time_z_max, self.acZ1_max, c='red')
            axs2[2].scatter(self.time_z_min, self.acZ1_min, c='green')
            axs2[2].legend()
            plt.tight_layout()

    def plot_results(self):
        plt.figure(20)
        plt.scatter(range(1,len(self.flex_stiffs)+1), self.flex_stiffs, c=self.result_color, s=2)
        # plt.xlabel('Stalk Number')
        plt.ylabel('Flexural Stiffness')

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
        for stalk in self.flex_stiffs:
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

    def clear_intermediate_data(self):
        self.strain_1 = None
        self.strain_b1 = None
        self.strain_2 = None
        self.strain_b2 = None
        self.strain_1_raw = None
        self.strain_b1_raw = None
        self.strain_2_raw = None
        self.strain_b2_raw = None
        self.force = None
        self.position = None
        self.forceDT = None
        self.positionDT = None
        self.forceDDT = None
        self.positionDDT = None
        self.interaction_indices = None
        self.stalk_force = None
        self.stalk_position = None
        self.stalk_time = None
        self.stalk_forces = None
        self.stalk_positions = None
        self.stalk_times = None
        self.force_fits = None
        self.position_fits = None

    def get_accel_coeff(self):
        cal_csv_path = r'AllInOne\accel_calibration_history.csv'
        cal_data = pd.read_csv(cal_csv_path)
        latest_cal = cal_data.iloc[-1]

        self.m_x = latest_cal['Gain X']; self.b_x = latest_cal['Offset X']
        self.m_y = latest_cal['Gain Y']; self.b_y = latest_cal['Offset Y']
        self.m_z = latest_cal['Gain Z']; self.b_z = latest_cal['Offset Z']

    def calc_accel_coeff(self):
        def calculate_gain_offset(max_val, min_val):
            gain = 2 / (max_val - min_val)
            offset = -(max_val + min_val) / (max_val - min_val)
            return gain, offset
        
        x_gain, x_offset = calculate_gain_offset(np.mean(self.acX1_max), np.mean(self.acX1_min))
        y_gain, y_offset = calculate_gain_offset(np.mean(self.acY1_max), np.mean(self.acZ1_min))
        z_gain, z_offset = calculate_gain_offset(np.mean(self.acZ1_max), np.mean(self.acZ1_min))
        
        from datetime import datetime
        csv_path = r'AllInOne\accel_calibration_history.csv'
        row_data = [datetime.now().strftime("%m_%d_%Y"), x_offset, y_offset, z_offset, x_gain, y_gain, z_gain]
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)

    def plot_angle(self):
        x_g = self.acX*self.m_x + self.b_x
        y_g = self.acY*self.m_y + self.b_y
        z_g = self.acZ*self.m_z + self.b_z
        # Calculate angles (in radians) about global x and y axes
        theta_x = np.arctan2(-y_g, np.sqrt(x_g**2 + z_g**2))  # Angle about global x-axis
        theta_y = np.arctan2(x_g, np.sqrt(y_g**2 + z_g**2))  # Angle about global y-axis

        self.pitch = np.degrees(theta_x)
        self.roll = np.degrees(theta_y)
        self.pitch_smooth = savgol_filter(self.pitch, 200, 2)
        self.roll_smooth = savgol_filter(self.roll, 200, 2)

        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].plot(self.time, self.pitch, label='pitch', linewidth=0.3)
        ax[0].plot(self.time, self.pitch_smooth, label='smooth')
        ax[0].axhline(0.05, c='red', linewidth=0.3)
        ax[0].axhline(3.65, c='red', linewidth=0.3)
        ax[0].axhline(-3.65, c='red', linewidth=0.3)
        ax[0].set_title(f'm:{self.m_x}, b:{self.b_x}')

        ax[0].legend()
        ax[1].plot(self.time, self.roll, label='roll', linewidth=0.3)
        ax[1].plot(self.time, self.roll_smooth, label='smooth')
        ax[1].axhline(-0.45, c='red', linewidth=0.3)
        ax[1].axhline(-2.8, c='red', linewidth=0.3)
        ax[1].axhline(2.8, c='red', linewidth=0.3)
        ax[1].set_title(f'm:{self.m_y}, b:{self.b_y}')
        ax[1].legend()

def boxplot_data(rodney_config, date=None, stalk_type=None, plot_num=20):
    results_df = pd.read_csv(r'Results\results.csv')
    
    if not date == None:
        results_df = results_df[results_df['Date'] == date]
    if not stalk_type == None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    offset = config_results['sensor offset (cm to gauge 2)'].iloc[0]
    print(offset)

    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    med_stats = med_EIs.describe()
    med_mean = np.mean(med_stats.loc['mean'])

    plt.figure(plot_num)
    plt.title(fr'{rodney_config}, Offset: {offset}cm, Avg: {med_mean:.2f} N/$m^2$')
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

def get_stats(rodney_config, date=None, stalk_type=None, plot_num=None):
    def get_ci(data, type_mean, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        # sem = stats.sem(data)
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
    lo_relMargins = np.array([lo_ci[col]['rel_margin'] for col in lo_EIs.columns])

    med_stats = med_EIs.describe()
    med_mean = np.mean(med_stats.loc['mean'])
    med_ci = {col: get_ci(med_EIs[col], med_mean) for col in med_EIs.columns}
    med_relMargins = np.array([med_ci[col]['rel_margin'] for col in med_EIs.columns])

    hi_stats = hi_EIs.describe()
    hi_mean = np.mean(hi_stats.loc['mean'])
    hi_ci = {col: get_ci(hi_EIs[col], hi_mean) for col in hi_EIs.columns}
    hi_relMargins = np.array([hi_ci[col]['rel_margin'] for col in hi_EIs.columns])

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
        plt.title(rodney_config + f', Median: {all_relMargins_median*100:.1f}')
        plt.legend()

    return all_relMargins_mean, all_relMargins_median

# Optimization functions
def get_all_tests(date):
    test_nums = []
    for file in os.listdir(f'Raw Data/{date}'):
        if file.endswith('.csv') and 'test' in file:
            test_num = file.split('_test_')[1].split('.csv')[0]
            test_nums.append(test_num)
    return test_nums

def get_config(date, test_num):
    csv_path = f'Raw Data/{date}/{date}_test_{test_num}.csv'
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "rodney configuration":
                return row[1]
    return None

def get_tests_for_config(date, rodney_config):
    tests = []
    for test_num in get_all_tests(date):
        config = get_config(date, test_num)
        if config == rodney_config:
            tests.append(test_num)
    return tests

# Helper function to load optimized parameters
def load_optimized_parameters(rodney_config):
    param_file = 'Results/optimized_parameters.csv'
    default_params = {
        'lo_pos_accel_tol': 1.0,
        'med_pos_accel_tol': 0.5,
        'hi_pos_accel_tol': 4.0,
        'lo_force_accel_tol': 25.0,
        'med_force_accel_tol': 30.0,
        'hi_force_accel_tol': 150.0
    }
    if os.path.isfile(param_file):
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            row = df[df['rodney_config'] == rodney_config].iloc[0]
            return {
                'lo_pos_accel_tol': float(row['lo_pos_accel_tol']),
                'med_pos_accel_tol': float(row['med_pos_accel_tol']),
                'hi_pos_accel_tol': float(row['hi_pos_accel_tol']),
                'lo_force_accel_tol': float(row['lo_force_accel_tol']),
                'med_force_accel_tol': float(row['med_force_accel_tol']),
                'hi_force_accel_tol': float(row['hi_force_accel_tol'])
            }
    return default_params

# Modified process_data to use optimized parameters
def process_data(date, test_num, view=False, overwrite=False):
    # Load optimized filtering parameters for the configuration
    config = get_config(date, test_num)
    params = load_optimized_parameters(config)
    
    test = LabStalkRow(date=date, test_num=test_num,
                       lo_pos_accel_tol=params['lo_pos_accel_tol'],
                       med_pos_accel_tol=params['med_pos_accel_tol'],
                       hi_pos_accel_tol=params['hi_pos_accel_tol'],
                       lo_force_accel_tol=params['lo_force_accel_tol'],
                       med_force_accel_tol=params['med_force_accel_tol'],
                       hi_force_accel_tol=params['hi_force_accel_tol'])
    test.smooth_strains(window=40, order=1)
    # test.correct_linear_drift()
    test.shift_initials()
    test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
    test.differentiate_force_position(smooth=True, window=40, order=1)
    test.differentiate_force_position_DT(smooth=True, window=40, order=1)
    test.find_stalk_interaction()
    test.collect_stalk_sections()
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

def optimize_parameters(dates, rodney_config):
    '''This optimizes the filtering paramters to provide the lowest average relative margin of error for all stalk types
    of a specified configuration. However, the median value is reported in other places. This is because using the 
    median as the objective score allows the optimizer to give bad results for troublesome data and effectively ignore
    it. In the worst case, it can choose values that result in non-detections of bad stalk interactions, whose results
    are recorded as np.nan and do not affect later statistics used to provide the objective score.
    
    There are 6 parameters to optimize. The optimizer samples the field with 30 combinations, and then selectively targets
    areas to find the optimal combination. This exploration takes a while, and notable imporovements don't happen until a 
    while into the process. This function will report the best value after each objective call, and after 120 combinations, 
    it will start counting how many calls were made without improvement, stopping after 30 calls with no improvement. So, 
    this will run at least 150 combinations, up to 200'''
    
    import gc
    from psutil import Process, HIGH_PRIORITY_CLASS
    from os import getpid

    inst = Process(getpid())
    inst.nice(HIGH_PRIORITY_CLASS)


    # Precompute data for all tests
    class PrecomputedTest:
        def __init__(self, date, test_num):
            test = LabStalkRow(date=date, test_num=test_num)
            test.smooth_strains(window=20, order=1)
            test.correct_linear_drift()
            test.shift_initials()
            test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
            test.differentiate_force_position(smooth=True, window=40, order=1)
            test.differentiate_force_position_DT(smooth=True, window=40, order=1)
            self.time = test.time
            self.force = test.force
            self.position = test.position
            self.forceDT = test.forceDT
            self.positionDT = test.positionDT
            self.forceDDT = test.forceDDT
            self.positionDDT = test.positionDDT
            self.stalk_type = test.stalk_type
            self.min_position = test.min_position
            self.max_position = test.max_position
            self.min_force = test.min_force
            self.min_force_rate = test.min_force_rate
            self.min_sequential = test.min_sequential
            self.height = test.height
            self.yaw = test.yaw
            self.results_path = test.results_path
            self.csv_path = test.csv_path
            self.header_rows = test.header_rows
    
    all_tests = []
    for date in dates:
        test_nums = get_tests_for_config(date, rodney_config)
        for test_num in test_nums:
            all_tests.append((date, test_num))
    print(f'Found {len(all_tests)} files for {rodney_config}')
    
    precomputed_tests = [PrecomputedTest(date, test_num) for date, test_num in all_tests]
    
    # Define the search space
    search_space = [
        Real(0.5, 3.5, name='lo_pos_accel_tol'),
        Real(0.5, 3.5, name='med_pos_accel_tol'),
        Real(1.0, 5.0, name='hi_pos_accel_tol'),
        Real(30, 70, name='lo_force_accel_tol'),
        Real(40, 100, name='med_force_accel_tol'),
        Real(40, 100, name='hi_force_accel_tol')
    ]
    
    # Objective function using precomputed data
    call = [0]
    scores = []
    best = [10]
    count = [0]
    def objective(params):
        call[0] += 1
        global results
        results.clear()
        lo_pos_accel_tol, med_pos_accel_tol, hi_pos_accel_tol, \
        lo_force_accel_tol, med_force_accel_tol, hi_force_accel_tol = params
        
        for precomp in precomputed_tests:
            # Extract date and test_num using os.path
            base_name = os.path.basename(precomp.csv_path)
            date = base_name.split('_test_')[0]
            test_num = base_name.split('_test_')[1].split('.csv')[0]
            
            test = LabStalkRow(date=date, test_num=test_num)
            test.time = precomp.time
            test.force = precomp.force
            test.position = precomp.position
            test.forceDT = precomp.forceDT
            test.positionDT = precomp.positionDT
            test.forceDDT = precomp.forceDDT
            test.positionDDT = precomp.positionDDT
            test.stalk_type = precomp.stalk_type
            test.min_position = precomp.min_position
            test.max_position = precomp.max_position
            test.min_force = precomp.min_force
            test.min_force_rate = precomp.min_force_rate
            test.min_sequential = precomp.min_sequential
            test.height = precomp.height
            test.yaw = precomp.yaw
            test.results_path = precomp.results_path
            test.csv_path = precomp.csv_path
            test.header_rows = precomp.header_rows
            
            if test.stalk_type == 'lo':
                test.pos_accel_tol = lo_pos_accel_tol
                test.force_accel_tol = lo_force_accel_tol
            elif test.stalk_type == 'med':
                test.pos_accel_tol = med_pos_accel_tol
                test.force_accel_tol = med_force_accel_tol
            elif test.stalk_type == 'hi':
                test.pos_accel_tol = hi_pos_accel_tol
                test.force_accel_tol = hi_force_accel_tol
            
            test.find_stalk_interaction()
            test.collect_stalk_sections()
            test.calc_stalk_stiffnesses()
            test.save_results(overwrite_result=True)
            test.clear_intermediate_data()
            del test
            gc.collect()
        
        score = get_stats(rodney_config)[0]
        scores.append(score)
        if call[0] > 120:
            count[0] += 1
        if score < best[0] and score > 0:
            best[0] = score
            count[0] = 0
        print(f"Parameters: {[f'{p:.3f}' for p in params]}, Score: {score:.6f}, Best: {best[0]:.6f}")
        return score
    
    # Early stopping callback
    def early_stopping(res):
        if len(scores) >= 30:
            print(call[0], count[0])
            if count[0] > 30 and call[0] >= 120:
                print("Early stopping: Score improvement too low")
                return True
        return False
    
    # Perform Bayesian optimization
    print("Starting Bayesian optimization")
    result = gp_minimize(objective, search_space, n_initial_points=30, n_calls=300, callback=[early_stopping])
    
    # Extract and save best parameters
    best_params = result.x
    best_score = result.fun
    param_file = 'Results/optimized_parameters.csv'
    headers = ['rodney_config', 'lo_pos_accel_tol', 'med_pos_accel_tol',
               'hi_pos_accel_tol', 'lo_force_accel_tol', 'med_force_accel_tol',
               'hi_force_accel_tol', 'best_score']
    row = [rodney_config] + best_params + [best_score]
    
    file_exists = os.path.isfile(param_file)
    if file_exists:
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            df.loc[df['rodney_config'] == rodney_config, headers] = row
            df.to_csv(param_file, index=False)
        else:
            with open(param_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)
    else:
        with open(param_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(row)
    
    print(f"Best parameters for {rodney_config}: "
          f"lo_pos_accel_tol={best_params[0]:.4f}, "
          f"med_pos_accel_tol={best_params[1]:.4f}, "
          f"hi_pos_accel_tol={best_params[2]:.4f}, "
          f"lo_force_accel_tol={best_params[3]:.4f}, "
          f"med_force_accel_tol={best_params[4]:.4f}, "
          f"hi_force_accel_tol={best_params[5]:.4f}, "
          f"Best score: {best_score:.6f}")

def optimize_parameters2(dates, rodney_config, stalk_type):
    from psutil import Process, HIGH_PRIORITY_CLASS
    from os import getpid
    import gc
    '''Optimizes filtering parameters for a specified stalk type to minimize the average relative margin of error.
    
    Args:
        dates: List of dates to process.
        rodney_config: Configuration identifier.
        stalk_type: Type of stalk to optimize ('lo', 'med', or 'hi').
    
    The function optimizes two parameters for the specified stalk type, using existing or default values for others.
    It updates only those two parameters and the corresponding score in optimized_parameters.csv.
    '''
    
    inst = Process(getpid())
    inst.nice(HIGH_PRIORITY_CLASS)

    # Precompute data for all tests
    class PrecomputedTest:
        def __init__(self, date, test_num):
            test = LabStalkRow(date=date, test_num=test_num)
            test.smooth_strains(window=20, order=1)
            test.correct_linear_drift()
            test.shift_initials()
            test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
            test.differentiate_force_position(smooth=True, window=40, order=1)
            test.differentiate_force_position_DT(smooth=True, window=40, order=1)
            self.time = test.time
            self.force = test.force
            self.position = test.position
            self.forceDT = test.forceDT
            self.positionDT = test.positionDT
            self.forceDDT = test.forceDDT
            self.positionDDT = test.positionDDT
            self.stalk_type = test.stalk_type
            self.min_position = test.min_position
            self.max_position = test.max_position
            self.min_force = test.min_force
            self.min_force_rate = test.min_force_rate
            self.min_sequential = test.min_sequential
            self.height = test.height
            self.yaw = test.yaw
            self.results_path = test.results_path
            self.csv_path = test.csv_path
            self.header_rows = test.header_rows
    
    all_tests = []
    for date in dates:
        test_nums = get_tests_for_config(date, rodney_config)
        for test_num in test_nums:
            all_tests.append((date, test_num))
    print(f'Found {len(all_tests)} files for {rodney_config}')
    
    precomputed_tests = [PrecomputedTest(date, test_num) for date, test_num in all_tests]
    
    # Define default parameters
    DEFAULT_POS_ACCEL_TOL = 2.0
    DEFAULT_FORCE_ACCEL_TOL = 50
    
    # Get existing parameters if available
    def get_existing_params(rodney_config):
        param_file = 'Results/optimized_parameters.csv'
        if os.path.isfile(param_file):
            df = pd.read_csv(param_file)
            if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
                row = df[df['rodney_config'] == rodney_config].iloc[0]
                return {
                    'lo_pos_accel_tol': row.get('lo_pos_accel_tol', np.nan),
                    'med_pos_accel_tol': row.get('med_pos_accel_tol', np.nan),
                    'hi_pos_accel_tol': row.get('hi_pos_accel_tol', np.nan),
                    'lo_force_accel_tol': row.get('lo_force_accel_tol', np.nan),
                    'med_force_accel_tol': row.get('med_force_accel_tol', np.nan),
                    'hi_force_accel_tol': row.get('hi_force_accel_tol', np.nan)
                }
        return {
            'lo_pos_accel_tol': DEFAULT_POS_ACCEL_TOL,
            'med_pos_accel_tol': DEFAULT_POS_ACCEL_TOL,
            'hi_pos_accel_tol': DEFAULT_POS_ACCEL_TOL,
            'lo_force_accel_tol': DEFAULT_FORCE_ACCEL_TOL,
            'med_force_accel_tol': DEFAULT_FORCE_ACCEL_TOL,
            'hi_force_accel_tol': DEFAULT_FORCE_ACCEL_TOL
        }
    
    existing_params = get_existing_params(rodney_config)
    
    # Define the search space based on stalk_type
    if stalk_type == 'lo':
        search_space = [
            Real(0.5, 3.5, name='lo_pos_accel_tol'),
            Real(30, 70, name='lo_force_accel_tol')
        ]
    elif stalk_type == 'med':
        search_space = [
            Real(0.5, 3.5, name='med_pos_accel_tol'),
            Real(40, 100, name='med_force_accel_tol')
        ]
    elif stalk_type == 'hi':
        search_space = [
            Real(1.0, 5.0, name='hi_pos_accel_tol'),
            Real(40, 100, name='hi_force_accel_tol')
        ]
    else:
        raise ValueError("Invalid stalk_type. Must be 'lo', 'med', or 'hi'.")
    
    # Objective function using precomputed data
    call = [0]
    scores = []
    best = [10]
    count = [0]
    def objective(params):
        call[0] += 1
        global results
        results.clear()
        
        for precomp in precomputed_tests:
            base_name = os.path.basename(precomp.csv_path)
            date = base_name.split('_test_')[0]
            test_num = base_name.split('_test_')[1].split('.csv')[0]
            
            test = LabStalkRow(date=date, test_num=test_num)
            test.time = precomp.time
            test.force = precomp.force
            test.position = precomp.position
            test.forceDT = precomp.forceDT
            test.positionDT = precomp.positionDT
            test.forceDDT = precomp.forceDDT
            test.positionDDT = precomp.positionDDT
            test.stalk_type = precomp.stalk_type
            test.min_position = precomp.min_position
            test.max_position = precomp.max_position
            test.min_force = precomp.min_force
            test.min_force_rate = precomp.min_force_rate
            test.min_sequential = precomp.min_sequential
            test.height = precomp.height
            test.yaw = precomp.yaw
            test.results_path = precomp.results_path
            test.csv_path = precomp.csv_path
            test.header_rows = precomp.header_rows
            
            if test.stalk_type == stalk_type:
                test.pos_accel_tol = params[0]
                test.force_accel_tol = params[1]
                test.find_stalk_interaction()
                test.collect_stalk_sections()
                test.calc_stalk_stiffnesses()
                test.save_results(overwrite_result=True)
            else:
                pos_key = f"{test.stalk_type}_pos_accel_tol"
                force_key = f"{test.stalk_type}_force_accel_tol"
                pos_tol = existing_params.get(pos_key, DEFAULT_POS_ACCEL_TOL)
                force_tol = existing_params.get(force_key, DEFAULT_FORCE_ACCEL_TOL)
                test.pos_accel_tol = pos_tol if not np.isnan(pos_tol) else DEFAULT_POS_ACCEL_TOL
                test.force_accel_tol = force_tol if not np.isnan(force_tol) else DEFAULT_FORCE_ACCEL_TOL
            
            
            test.clear_intermediate_data()
            del test
            gc.collect()
        
        score = get_stats2(rodney_config, stalk_type=stalk_type)[0]
        scores.append(score)
        if call[0] > 20:
            count[0] += 1
        if score < best[0]:
            best[0] = score
            count[0] = 0
        print(f"Parameters: {[f'{p:.3f}' for p in params]}, Score: {score:.6f}, Best: {best[0]:.6f}")
        return score
    
    # Early stopping callback (corrected to stop after 30 calls without improvement)
    def early_stopping(res):
        print(call[0], count[0])
        if len(scores) >= 20:
            if count[0] > 20 and call[0] >= 50:
                print("Early stopping: Score improvement too low")
                return True
        return False
    
    # Perform Bayesian optimization
    print(f"Starting Bayesian optimization for {stalk_type}")
    result = gp_minimize(objective, search_space, n_initial_points=20, n_calls=200, callback=[early_stopping])
    
    # Extract and save best parameters
    best_params = result.x
    best_score = result.fun
    param_file = 'Results/optimized_parameters.csv'
    headers = ['rodney_config', 'lo_pos_accel_tol', 'med_pos_accel_tol', 'hi_pos_accel_tol',
               'lo_force_accel_tol', 'med_force_accel_tol', 'hi_force_accel_tol',
               'lo_score', 'med_score', 'hi_score']
    
    file_exists = os.path.isfile(param_file)
    if file_exists:
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            idx = df[df['rodney_config'] == rodney_config].index[0]
            if stalk_type == 'lo':
                df.at[idx, 'lo_pos_accel_tol'] = best_params[0]
                df.at[idx, 'lo_force_accel_tol'] = best_params[1]
                df.at[idx, 'lo_score'] = best_score
            elif stalk_type == 'med':
                df.at[idx, 'med_pos_accel_tol'] = best_params[0]
                df.at[idx, 'med_force_accel_tol'] = best_params[1]
                df.at[idx, 'med_score'] = best_score
            elif stalk_type == 'hi':
                df.at[idx, 'hi_pos_accel_tol'] = best_params[0]
                df.at[idx, 'hi_force_accel_tol'] = best_params[1]
                df.at[idx, 'hi_score'] = best_score
            df.to_csv(param_file, index=False)
        else:
            new_row = pd.DataFrame([[rodney_config] + [np.nan]*9], columns=headers)
            if stalk_type == 'lo':
                new_row.at[0, 'lo_pos_accel_tol'] = best_params[0]
                new_row.at[0, 'lo_force_accel_tol'] = best_params[1]
                new_row.at[0, 'lo_score'] = best_score
            elif stalk_type == 'med':
                new_row.at[0, 'med_pos_accel_tol'] = best_params[0]
                new_row.at[0, 'med_force_accel_tol'] = best_params[1]
                new_row.at[0, 'med_score'] = best_score
            elif stalk_type == 'hi':
                new_row.at[0, 'hi_pos_accel_tol'] = best_params[0]
                new_row.at[0, 'hi_force_accel_tol'] = best_params[1]
                new_row.at[0, 'hi_score'] = best_score
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(param_file, index=False)
    else:
        new_row = pd.DataFrame([[rodney_config] + [np.nan]*9], columns=headers)
        if stalk_type == 'lo':
            new_row.at[0, 'lo_pos_accel_tol'] = best_params[0]
            new_row.at[0, 'lo_force_accel_tol'] = best_params[1]
            new_row.at[0, 'lo_score'] = best_score
        elif stalk_type == 'med':
            new_row.at[0, 'med_pos_accel_tol'] = best_params[0]
            new_row.at[0, 'med_force_accel_tol'] = best_params[1]
            new_row.at[0, 'med_score'] = best_score
        elif stalk_type == 'hi':
            new_row.at[0, 'hi_pos_accel_tol'] = best_params[0]
            new_row.at[0, 'hi_force_accel_tol'] = best_params[1]
            new_row.at[0, 'hi_score'] = best_score
        new_row.to_csv(param_file, index=False)
    
    print(f"Best parameters for {rodney_config} ({stalk_type}): "
          f"{stalk_type}_pos_accel_tol={best_params[0]:.4f}, "
          f"{stalk_type}_force_accel_tol={best_params[1]:.4f}, "
          f"Best score: {best_score:.6f}")

def get_stats2(rodney_config, date=None, stalk_type=None, plot_num=None):
    '''Computes statistics for stalk stiffness measurements.
    
    Args:
        rodney_config: Configuration identifier.
        date: Optional date to filter results.
        stalk_type: Optional stalk type to filter results ('lo', 'med', or 'hi').
        plot_num: Optional plot number for visualization.
    
    Returns:
        Tuple of mean and median of relative margins of error.
    '''
    def get_ci(data, type_mean, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std / np.sqrt(n))
        margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * std / np.sqrt(n)
        rel_margin = margin / type_mean if type_mean != 0 else 0
        return {'interval': ci, 'mean': mean, 'margin': margin, 'rel_margin': rel_margin}
    
    results_df = pd.read_csv(r'Results\results.csv')
    if date is not None:
        results_df = results_df[results_df['Date'] == date]
    if stalk_type is not None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    
    if config_results.empty:
        return np.nan, np.nan
    
    EIs = config_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 
                          'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    stats_df = EIs.describe()
    mean_EI = np.mean(stats_df.loc['mean'])
    ci_dict = {col: get_ci(EIs[col], mean_EI) for col in EIs.columns}
    rel_margins = np.array([ci_dict[col]['rel_margin'] for col in EIs.columns])
    
    all_relMargins_mean = np.nanmean(rel_margins)
    all_relMargins_median = np.nanmedian(rel_margins)
    
    if plot_num is not None:
        plt.figure(plot_num)
        plt.scatter(range(1, len(rel_margins)+1), rel_margins*100, 
                    label=stalk_type, 
                    c='red' if stalk_type=='lo' else 'green' if stalk_type=='med' else 'blue')
        plt.axhline(all_relMargins_median*100, c='black', label='Median')
        plt.ylim(0, 80)
        plt.ylabel('% Relative Error Margin')
        plt.title(f'{rodney_config} ({stalk_type}), Median: {all_relMargins_median*100:.1f}')
        plt.legend()
    
    return all_relMargins_mean, all_relMargins_median

def show_force_position(dates, test_nums, stalk_id=None, rodney_config=None):
    for date in dates:
        for test_num in test_nums:
            config = get_config(date, test_num)
            params = load_optimized_parameters(config)
            test = LabStalkRow(date=date, test_num=test_num,
                       lo_pos_accel_tol=params['lo_pos_accel_tol'],
                       med_pos_accel_tol=params['med_pos_accel_tol'],
                       hi_pos_accel_tol=params['hi_pos_accel_tol'],
                       lo_force_accel_tol=params['lo_force_accel_tol'],
                       med_force_accel_tol=params['med_force_accel_tol'],
                       hi_force_accel_tol=params['hi_force_accel_tol'])
            test.smooth_strains(window=20, order=1)
            if test.accel_flag:
                test.smooth_accels(window=50, order=1)
                test.differentiate_strains()
                test.smooth_strain_DTs(window=1000, order=2)
            test.correct_linear_drift()
            test.shift_initials()
            test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
            test.plot_force_position(view_stalks=False, plain=True)
            test.plot_raw_strain()
            if test.accel_flag:
                test.get_accel_coeff()
                # test.calc_accel_coeff()
                test.plot_angle()
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
    plt.xlabel('Darling Stiffness')
    plt.ylabel('Rodney Stiffness')
    plt.axis('equal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    local_run_flag = True
    
    '''Batch run of same configuration'''
    for i in range(61, 70+1):
        process_data(date='08_19', test_num=f'{i}', view=False, overwrite=True)
    # show_force_position(dates=['08_06'], test_nums=[1])

    # boxplot_data(rodney_config='Integrated Beam Prototype 1', date='07_03', plot_num=104)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', date='07_10', plot_num=105)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', date='07_14', plot_num=106)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', date='07_10', plot_num=107)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', date='07_11', plot_num=108)
    # boxplot_data(rodney_config='Integrated Beam Printed Guide 1', date='07_16', plot_num=109)
    # boxplot_data(rodney_config='Integrated Beam Fillet 1', date='08_13', plot_num=110)

    boxplot_data(rodney_config='Integrated Beam Fillet 1', date='08_19', plot_num=108)
    '''end batch run'''

    '''Statistics'''
    # print('1 mean, median', get_stats(rodney_config='Integrated Beam Prototype 1', plot_num=204))
    # print('2 mean, median', get_stats(rodney_config='Integrated Beam Prototype 2', date='07_10', plot_num=205))
    # print('2 mean, median', get_stats(rodney_config='Integrated Beam Prototype 2', date='07_14', plot_num=206))
    # print('3 mean, median', get_stats(rodney_config='Integrated Beam Prototype 3', date='07_10', plot_num=207))
    # print('3 mean, median', get_stats(rodney_config='Integrated Beam Prototype 3', date='07_11', plot_num=208))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Printed Guide 1', date='07_16', plot_num=209))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Fillet 1', date='07_24', plot_num=210))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Fillet 1', date='08_13', plot_num=211))
    '''end statistics'''

    '''Single file run and view full file. Does not save result'''
    # process_data(date='07_14', test_num='1', view=True)
    '''end single file run'''

    # Optimize parameters for a specific configuration
    # optimize_parameters(dates=['07_16'], rodney_config='Integrated Beam Printed Guide 1')

    # correlation(rodney_config='Integrated Beam Fillet 1', date='08_13')

    plt.show()