import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
import scipy.stats as stats
import csv
import os
from scipy.signal import savgol_filter
from scipy.stats import linregress
import json

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
        self.height = section.height
        self.yaw = section.yaw
        self.B = section.max_position

        # Force vs deflection
        self.alpha = np.radians(32.1)
        self.pos_x = (self.B - self.position)*np.sin(self.yaw)/np.cos(self.alpha)
        # self.force = self.force/np.cos(self.yaw - self.alpha)

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

    def interactive_calc_stiffness(self, stalk_num, test_num, date, stalk_type):
        # Discard all position = 0
        mask = self.position != 0
        self.time = self.time[mask]
        self.force = self.force[mask]
        self.position = self.position[mask]
        self.pos_x = self.pos_x[mask]

        # Keep indices where x is strictly decreasing over reverse time
        keep_indices = [len(self.pos_x) - 1]
        for i in range(len(self.pos_x) - 2, -1, -1):
            if self.pos_x[i] < self.pos_x[keep_indices[-1]]:
                keep_indices.append(i)
        
        keep_indices.reverse()
        if len(keep_indices) < 10:
            self.stiffness = np.nan
            return
        
        self.time = np.array([self.time[i] for i in keep_indices])
        self.pos_x = np.array([self.pos_x[i] for i in keep_indices])
        self.force = np.array([self.force[i] for i in keep_indices])
        self.position = np.array([self.position[i] for i in keep_indices])

        # Create interactive plot with complete data
        fig, ax = plt.subplots(2, 2, sharex=False, figsize=(9, 9))#, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.setGeometry(100, 100, 900, 900)
        ax[1, 0].remove()
        ax[1, 1].remove()
        ax_bottom = fig.add_subplot(212, sharex=None, sharey=None) # Create a new full-width subplot for bottom row
        # Gradient color based on time
        indices = np.arange(len(self.time))
        norm = plt.Normalize(indices.min(), indices.max())
        cmap = plt.get_cmap('viridis')

        ax[0,0].scatter(self.time, self.force, s=5, c=indices, cmap=cmap, norm=norm)
        ax[0,0].set_ylabel('Force (N)')
        ax[0,0].set_xlabel('Time (s)')

        ax[0,1].scatter(self.time, self.position, s=5, c=indices, cmap=cmap, norm=norm)
        # ax[0,1].set_ylim(0, 0.20)
        ax[0,1].set_ylabel('Position (m)')
        ax[0,1].set_xlabel('Time (s)')

        ax_bottom.scatter(self.pos_x, self.force, s=5, c=indices, cmap=cmap, norm=norm)
        # ax_bottom.set_xlim(0, 0.05)
        ax_bottom.set_ylabel('Force (N)')
        ax_bottom.set_xlabel('Deflection (m)')

        plt.suptitle(f'Select spans on FvX for stiffness calculation\nStalk:{stalk_num}, Test:{test_num}')

        self.selected_spans = []
        self.current_span = None
        self.current_span_patches = {'force_pos': None, 'force_time': None, 'pos_time': None}

        def onselect(xmin, xmax):
            # Clear previous spans
            for patch in self.current_span_patches.values():
                if patch:
                    patch.remove()

            self.current_span = (xmin, xmax)
            # Convert position range to time range for force vs time and position vs time plots
            time_mask = (self.pos_x >= xmin) & (self.pos_x <= xmax)
            if np.any(time_mask):
                time_min, time_max = np.min(self.time[time_mask]), np.max(self.time[time_mask])
                # Draw spans on all plots
                self.current_span_patches['force_time'] = ax[0, 0].axvspan(time_min, time_max, alpha=0.5, facecolor='tab:blue')
                self.current_span_patches['pos_time'] = ax[0, 1].axvspan(time_min, time_max, alpha=0.5, facecolor='tab:blue')
                self.current_span_patches['force_pos'] = ax_bottom.axvspan(xmin, xmax, alpha=0.5, facecolor='tab:blue')
            fig.canvas.draw_idle()

        span = SpanSelector(ax_bottom, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='tab:blue'))

        def add(event):
            if self.current_span:
                self.selected_spans.append(self.current_span)
                # Convert position range to time range for green spans
                time_mask = (self.pos_x >= self.current_span[0]) & (self.pos_x <= self.current_span[1])
                if np.any(time_mask):
                    time_min, time_max = np.min(self.time[time_mask]), np.max(self.time[time_mask])
                    # Draw green spans on all plots
                    ax[0, 0].axvspan(time_min, time_max, alpha=0.3, facecolor='green')
                    ax[0, 1].axvspan(time_min, time_max, alpha=0.3, facecolor='green')
                    ax_bottom.axvspan(self.current_span[0], self.current_span[1], alpha=0.3, facecolor='green')
                # Clear current span
                for patch in self.current_span_patches.values():
                    if patch:
                        patch.remove()
                self.current_span = None
                self.current_span_patches = {'force_pos': None, 'force_time': None, 'pos_time': None}
                fig.canvas.draw_idle()

        def done(event):
            if self.selected_spans:
                mask = np.full(len(self.pos_x), False)
                for xmin, xmax in self.selected_spans:
                    current_mask = (self.pos_x >= xmin) & (self.pos_x <= xmax)
                    mask = np.logical_or(mask, current_mask)
                selected_pos_x = self.pos_x[mask]
                selected_force = self.force[mask]
                selected_pos = self.position[mask]
                selected_time = self.time[mask]
                save_stalk(selected_time, selected_force, selected_pos)
                if len(selected_pos_x) >= 2:
                    slope, intercept, r, _, _ = stats.linregress(selected_pos_x, selected_force)
                    self.slope_fx = slope
                    self.intercept_fx = intercept
                    self.r_fx = r
                    self.stiffness = (slope * self.height**3) / 3
                else:
                    self.stiffness = np.nan
            else:
                self.stiffness = np.nan
            plt.close(fig)

        def reject(event):
            self.stiffness = np.nan
            plt.close(fig)

        def save_stalk(time, force, position):
            os.makedirs(f'Results/Field/{date}/{stalk_type}/Stalk Traces', exist_ok=True)
            df = pd.DataFrame({'Time': time, 'Force': force, 'Position': position})
            path = f'Results/Field/{date}/{stalk_type}/Stalk Traces/S{stalk_num:02d}_{test_num:02d}.csv'
            df.to_csv(path, index=False)
            print(f"Saved stalk {stalk_num} to {path}")  # Debug output

        ax_add = plt.axes([0.6, 0.025, 0.1, 0.075])
        btn_add = Button(ax_add, 'Add Span')
        btn_add.on_clicked(add)

        ax_done = plt.axes([0.7, 0.025, 0.1, 0.075])
        btn_done = Button(ax_done, 'Done')
        btn_done.on_clicked(done)

        ax_reject = plt.axes([0.8, 0.025, 0.1, 0.075])
        btn_reject = Button(ax_reject, 'Reject')
        btn_reject.on_clicked(reject)

        plt.show(block=True)

    def recalc_stiffness(self, stalk_num, test_num, date, stalk_type):
        path = f'Results/Field/{date}/{stalk_type}/Stalk Traces/S{stalk_num:02d}_{test_num:02d}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.time = df['Time'].to_numpy()
            self.force = df['Force'].to_numpy()
            self.position = df['Position'].to_numpy()

            self.pos_x = (self.B - self.position)*np.sin(self.yaw)/np.cos(self.alpha)
            # self.force = self.force/np.cos(self.yaw - self.alpha)
            
            if len(self.pos_x) >= 2:
                slope, intercept, r, _, _ = stats.linregress(self.pos_x, self.force)
                self.slope_fx = slope
                self.intercept_fx = intercept
                self.r_fx = r
                self.stiffness = (slope * self.height**3) / 3
            else:
                self.stiffness = np.nan
        else:
            self.stiffness = np.nan


class FieldStalkSection:
    def __init__(self, date, test_num, min_force_rate=-0.5, pos_accel_tol=0.8, force_accel_tol=700):
        # These params set the filter bounds for identifying which portions of the data are stalk interactions. These are ideally straight lines, increasing in
        # force and decreasing in position. 
            # this window should be a bit wider than the physical sensor
        self.min_position = 5*1e-2  # centimeters, location of 2nd strain gauge
        self.max_position = 18*1e-2 # centimeters, location of beam end
        self.min_force = 1 # newton, easily cut out the spaces between stalks. Rejects noisy detection regime
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
        self.results_path = os.path.join(parent_folder, r'field_results.csv')   # one file to store all results from all dates/sections
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
                if row[0] == "rodney configuration":
                    self.configuration = row[1]
                    params_read += 1
                if row[0] == "sensor calibration (k d c)":
                        # c values are not used, instead calculated from initial idle values of each data collection
                        # read in the strings and convert to floats
                    k_str = row[1]
                    d_str = row[2]
                    k_values = [float(v) for v in k_str.split()]
                    d_values = [float(v) for v in d_str.split()]
                    self.k_1, self.k_B1, self.k_2, self.k_B2 = k_values
                    self.d_1, self.d_B1, self.d_2, self.d_B2 = d_values
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
        self.strain_2 = self.strain_2_raw = data['Strain A2'].to_numpy()    # only uses the two channels
        mask = (self.time >= 0.001) & (self.time <= 600) & (np.diff(self.time, prepend=self.time[0]) >= 0)
        self.time = self.time[mask]
        self.strain_1 = self.strain_1_raw = self.strain_1_raw[mask]
        self.strain_2 = self.strain_2_raw = self.strain_2_raw[mask]
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
        if self.accel_flag:
            self.acX = savgol_filter(self.acX, accel_window, accel_order)
            self.acY = savgol_filter(self.acY, accel_window, accel_order)
            self.acZ = savgol_filter(self.acZ, accel_window, accel_order)

    def differentiate_accels(self, smooth=False, window = 1000, order=2):
        self.acX_DT = np.gradient(self.acX, self.time)
        self.acY_DT = np.gradient(self.acY, self.time)
        self.acZ_DT = np.gradient(self.acZ, self.time)
        if smooth:
            self.smooth_accel_DTs(window, order)
        
    def smooth_accel_DTs(self, window=1000, order=2):
        self.acX_DT = savgol_filter(self.acX_DT, window, order)
        self.acY_DT = savgol_filter(self.acY_DT, window, order)
        self.acZ_DT = savgol_filter(self.acZ_DT, window, order)

    def shift_initials(self, time_cutoff):
        cutoff_index = np.where(self.time - self.time[0] > time_cutoff)[0][0]
        self.strain_1_ini  = np.mean(self.strain_1[0:cutoff_index])
        self.strain_2_ini = np.mean(self.strain_2[0:cutoff_index]) 
        self.c_1 = self.strain_1_ini
        self.c_2 = self.strain_2_ini

    def calc_force_position(self, smooth=True, window=20, order=1, small_den_cutoff=5*1e-5):
        self.force_num = self.k_2*(self.strain_1 - self.c_1) - self.k_1*(self.strain_2 - self.c_2)
        self.force_den = self.k_1*self.k_2*(self.d_2 - self.d_1)
        self.force = self.force_raw = np.where(self.force_num / self.force_den < 0, 0, self.force_num / self.force_den)

        self.pos_num = self.k_2*self.d_2*(self.strain_1 - self.c_1) - self.k_1*self.d_1*(self.strain_2 - self.c_2)
        self.pos_den = self.k_2*(self.strain_1 - self.c_1) - self.k_1*(self.strain_2 - self.c_2)
        self.position = self.position_raw = np.clip(np.where(np.abs(self.pos_den) < small_den_cutoff, 0, self.pos_num/self.pos_den), 0, 0.30)

        if smooth:
            self.force = savgol_filter(self.force, window, order)
            self.position = savgol_filter(self.position, window, order)

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
        ax[0].set_ylabel('Force (N)')
        ax[1].plot(self.time - time_ini, self.position*100, label='Position')
        ax[0].plot(self.time - time_ini, self.force_raw, label='raw', linewidth=0.5)
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

    def calc_section_stiffnesses(self):
        for stalk in self.stalks:
            if not np.isnan(stalk.time).all():
                stalk.calc_stalk_stiffness()

    def plot_section_stiffnesses(self):
        try:
            time_ini = self.stalks[0].time_loc
            time_end = self.stalks[-1].time_loc
        except:
            time_ini = 0
            time_end = 10
        
        stalk_times = [stalk.time_loc - time_ini for stalk in self.stalks]
        stalk_stiffs = [stalk.stiffness for stalk in self.stalks]
        plt.figure(100)
        plt.scatter(stalk_times, stalk_stiffs, s=10)
        plt.xlabel('Time after first stalk (s)')
        plt.ylabel('Flexural Stiffness (N/m^2)')

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

    def interactive_clip_and_save(self, num_stalks):
        # Run automatic detection first to estimate num_stalks and guide user with red overlays
        self.find_stalk_interaction()
        self.collect_stalks()

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
        ax[0].plot(self.time, self.force, label='Force')
        ax[0].set_ylabel('Force (N)')
        ax[1].plot(self.time, self.position, label='Position')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (m)')
        plt.suptitle(f'{self.date} Test {self.test_num} {self.stalk_type}\nSelect {num_stalks} stalk interactions')

        # Overlay automatic detections in red for guidance
        for stalk in self.stalks:
            if not np.isnan(stalk.time).all():
                ax[0].plot(stalk.time, stalk.force, c='red')
                ax[1].plot(stalk.time, stalk.position, c='red')

        self.spans = []
        self.current_span = None
        self.current_span_patch = None

        def onselect(xmin, xmax):
            # Remove previous temporary span if it exists
            if self.current_span_patch:
                self.current_span_patch.remove()
            self.current_span = (xmin, xmax)
            self.current_span_patch = ax[1].axvspan(xmin, xmax, alpha=0.5, color='tab:blue')
            fig.canvas.draw_idle()
            print(self.current_span)

        self.span = SpanSelector(ax[1], onselect, 'horizontal', useblit=True,
                                props=dict(alpha=0.5, facecolor='tab:blue'))

        def confirm(event):
            print("Button")
            if self.current_span:
                print(f"Confirming span: {self.current_span}")  # Debug output
                self.spans.append(self.current_span)
                if self.current_span_patch:
                    self.current_span_patch.remove()
                ax[1].axvspan(self.current_span[0], self.current_span[1], alpha=0.3, color='green')
                self.current_span = None
                self.current_span_patch = None
                fig.canvas.draw_idle()
                if len(self.spans) == num_stalks:
                    save_stalks()
                    plt.close(fig)
            else:
                print("No span selected to confirm")  # Debug output

        def save_stalks():
            os.makedirs(f'Results/Field/{self.date}/{self.stalk_type}/Stalk Clips', exist_ok=True)
            for i, (tmin, tmax) in enumerate(self.spans):
                mask = (self.time >= tmin) & (self.time <= tmax)
                time_clip = self.time[mask]
                force_clip = self.force[mask]
                position_clip = self.position[mask]
                df = pd.DataFrame({'Time': time_clip, 'Force': force_clip, 'Position': position_clip})
                path = f'Results/Field/{self.date}/{self.stalk_type}/Stalk Clips/S{i+1:02d}_{self.test_num:02d}.csv'
                df.to_csv(path, index=False)
                print(f"Saved stalk {i+1} to {path}")  # Debug output

        # Create Confirm button
        ax_confirm = plt.axes([0.8, 0.025, 0.1, 0.075])
        btn_confirm = Button(ax_confirm, 'Confirm')
        btn_confirm.on_clicked(confirm)
        def on_key(event):
            if event.key == 'enter':
                confirm(event)

        # # Ensure the figure remains interactive
        fig.canvas.mpl_connect('close_event', lambda event: print(f"Closed window for Test {self.test_num}"))  # Debug
        fig.canvas.mpl_connect('key_press_event', on_key)
        # plt.ion()  # Enable interactive mode
        plt.show()


class TestResults:
    def __init__(self):
        self.tests = []
        self.groups = []

    def add_test(self, test):
        time_ini = test.stalks[0].time_loc
        test_data = {
            'date': test.date,
            'test_num': test.test_num,
            'stalk_type': test.stalk_type,
            'height': test.height,
            'yaw': test.yaw,
            'stalks': [{'time_loc': stalk.time_loc - time_ini,
                        'stiffness': stalk.stiffness} for stalk in test.stalks],
            'num_stalks': len(test.stalks)
        }
        self.tests.append(test_data)

    def get_all_stalks(self):
        return [stalk for test in self.tests for stalk in test['stalks']]

    def save_groups(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.groups, f)

    def load_groups(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.groups = json.load(f)
                self.groups.sort(key=lambda x: x['avg_time_loc'])

    def group_stalks_by_time(self, window_tol=1, date=None, stalk_type=None):
        all_stalks = self.get_all_stalks()
        if not all_stalks:
            print("No stalks in runtime. Trying load from file")
            
        filename = fr"Results\groups_{date}_{stalk_type}.json"
        self.load_groups(filename)
        if self.groups:
            print(f"Loaded groups from {filename}")
            return

        times = np.array([stalk['time_loc'] for stalk in all_stalks])
        fig, ax = plt.subplots()
        colors = ['blue'] * len(times)
        scatter = ax.scatter(range(len(times)), times, c=colors, picker=5)

        selected = set()
        processed = set()

        def update_colors():
            new_colors = ['blue'] * len(times)
            for i in selected:
                new_colors[i] = 'yellow'
            for i in processed:
                new_colors[i] = 'green'
            scatter.set_facecolors(new_colors)
            fig.canvas.draw_idle()

        def on_pick(event):
            if event.artist != scatter:
                return
            ind = event.ind[0]
            if ind in processed:
                return
            if ind in selected:
                selected.remove(ind)
            else:
                selected.add(ind)
            update_colors()

        def on_next(event):
            if selected:
                group_stiffness = [all_stalks[i]['stiffness'] for i in selected]
                group_times = [all_stalks[i]['time_loc'] for i in selected]
                self.groups.append({
                    'stiffnesses': group_stiffness,
                    'avg_time_loc': np.mean(group_times)
                })
                self.groups.sort(key=lambda x: x['avg_time_loc'])
                processed.update(selected)
                selected.clear()
                update_colors()

        def on_stop(event):
            if selected:
                on_next(event)
            self.save_groups(filename)
            plt.close(fig)

        fig.canvas.mpl_connect('pick_event', on_pick)

        ax_next = plt.axes([0.7, 0.05, 0.1, 0.075])
        btn_next = Button(ax_next, 'Next Stalk')
        btn_next.on_clicked(on_next)

        ax_stop = plt.axes([0.81, 0.05, 0.1, 0.075])
        btn_stop = Button(ax_stop, 'Stop')
        btn_stop.on_clicked(on_stop)

        plt.show()

    def show_results(self, correlation_flag, section):
        rodney_medians = []
        rodney_times = []
        for stalk in self.groups:
            rodney_medians.append(np.clip(np.median(stalk['stiffnesses']), 0, 30))
            plt.figure(1000)
            ele1= plt.boxplot(stalk['stiffnesses'], positions=[round(stalk['avg_time_loc'],1)], label='Rodney Boxplot')
            rodney_times.append(round(stalk['avg_time_loc'],1))
            
        
        if correlation_flag:
            darling_results = pd.read_csv(r'Results\Darling Field Data_08_07_2025.csv')[section.split()[0]].dropna()
            print(darling_results)
            darling_medians = [res for res in darling_results]
            print(darling_medians)
            print(rodney_medians)

            rodney_medians = np.array(rodney_medians)
            rodney_times = np.array(rodney_times)
            darling_medians = np.array(darling_medians)
            slope, inter, r, _, _ = linregress(darling_medians, rodney_medians)

            plt.figure(1000)
            ele2 = plt.scatter(rodney_times, darling_medians, label='Darling')
            ele3 = plt.scatter(rodney_times, rodney_medians, label='Rodney')
            plt.xlabel('Elapsed Test Time (s)')
            plt.ylabel('Median Stiffness (N/m^2)')
            plt.legend(handles=[ele1, ele2, ele3], labels=['Rodney Boxplot', 'Darling', 'Rodney'])

            plt.figure(1001)
            plt.plot(darling_medians, slope*darling_medians + inter, c='black', linewidth=0.5, label='Correlation Trendline')
            plt.scatter(darling_medians, rodney_medians, label=fr'Median $R^2$: {r**2:.4f} Slope: {slope:.3f}')

            plt.plot(darling_medians, darling_medians, c='blue', linewidth='0.5', label='1:1')
            plt.xlabel('Darling Stiffness')
            plt.ylabel('Rodney Stiffness')
            plt.axis('equal')
            plt.legend()
            plt.show()

        plt.show()

    def show_results_interactive(self, filepath, correlation_flag=True):
        # Normalize filepath and extract components
        filepath = os.path.normpath(filepath); parts = filepath.split(os.sep)
        date = parts[2]; stalk_type = parts[3]

        # Construct the CSV file path
        d_path = os.path.join(filepath, f'darling_{date}_{stalk_type}.csv')
        r_path = os.path.join(filepath, f'stiffness_{date}_{stalk_type}.csv')
        
        # Read the CSV files
        darling_df = pd.read_csv(d_path, index_col=0); rodney_df = pd.read_csv(r_path, index_col=0)
        stalks_results = [rodney_df.loc[stalk].dropna().to_numpy() for stalk in rodney_df.index]
        
        plt.figure()
        for i, stalk in enumerate(stalks_results):
            print(stalk)
            plt.boxplot(stalk, positions=[i])

        if correlation_flag:
            d_means = darling_df['Mean']; d_medians = darling_df['Median']; d_stds = darling_df['Std_Dev']
            r_means = rodney_df['Mean']; r_medians = rodney_df['Median']; r_stds = rodney_df['Std_Dev']
            
            slope, inter, r, _, _ = linregress(d_medians, r_medians)
            
            plt.figure()
            plt.scatter(d_medians, r_medians)
            plt.plot(d_medians, d_medians, c='black', linewidth=0.5)
            plt.plot(d_medians, slope*np.array(d_medians) + inter, c='orange', linewidth='0.5')
            plt.title(f'Date: {date}, Section: {stalk_type}\n'+ rf'$R^2$: {r**2:.4f}, Slope: {slope:.3f}')
            plt.axis('equal')
    

# Automatic processing
def show_force_position(dates, test_nums, show_accels):
    for date in dates:
        for test_num in test_nums:
            test = FieldStalkSection(date=date, test_num=test_num)
            if test.exist:
                test.smooth_raw_data()
                test.shift_initials(time_cutoff=1.0)
                test.calc_force_position()
                test.differentiate_force_position()
                test.differentiate_force_position_DT()
                test.find_stalk_interaction()
                test.collect_stalks()
                test.calc_section_stiffnesses()
                test.calc_angles()

                test.plot_force_position(view_stalks=True, show_accels=show_accels)
                test.plot_section_stiffnesses()
    # plt.show()

def show_accels(dates, test_nums):
    for date in dates:
        for test_num in test_nums:
            test = FieldStalkSection(date=date, test_num=test_num)
            if test.exist:
                test.smooth_raw_data()
                test.calc_angles()
                test.plot_accels()
    plt.show()

def process_and_store_section(dates, test_nums):
    sect_res = TestResults()
    for date in dates:
        for test_num in test_nums:
            test = FieldStalkSection(date=date, test_num=test_num)
            if test.exist:
                test.smooth_raw_data()
                test.shift_initials(time_cutoff=1.0)
                test.calc_force_position()
                test.differentiate_force_position()
                test.differentiate_force_position_DT()
                test.find_stalk_interaction()
                test.collect_stalks()
                test.calc_section_stiffnesses()
                test.calc_angles()

                sect_res.add_test(test)

        sect_res.group_stalks_by_time(date=date, stalk_type=test.stalk_type)

def show_section_results(dates, test_nums, correlation_flag=False):
    sect_res = TestResults()
    for date in dates:
        test = FieldStalkSection(date=date, test_num=test_nums[0])
        if test.exist:
            sect_res.group_stalks_by_time(date=date, stalk_type=test.stalk_type)
            sect_res.show_results(correlation_flag, test.stalk_type)

def show_day_results(date, correlation_flag=False):
    import re
    sections = []
    folder = r'Results'
    for filename in os.listdir(folder):
        if filename.endswith(".json") and date in filename:
            section_code = re.search(r"\d{2}-[A-Z]", filename).group()
            darling_results = pd.read_csv(rf'Results\Darling Field Data_{date}_2025.csv')[section_code].dropna()
            darling_medians = [res for res in darling_results]
            with open(os.path.join(folder, filename), 'r') as f:
                stalks = json.load(f)
                stalks.sort(key=lambda x: x['avg_time_loc'])
                rodney_medians = [np.median(stalk['stiffnesses']) for stalk in stalks]
                section = {'section_code': section_code, 'stalks': stalks, 'rodney_medians': rodney_medians, 'darling_medians': darling_medians}
                sections.append(section)
    max = 0
    all_darling = []; all_rodney = []
    for section in sections:
        section_max = np.max(section['darling_medians'])
        if section_max > max:
            max = section_max
        for val1, val2 in zip(section['darling_medians'], section['rodney_medians']):
            all_darling.append(val1); all_rodney.append(val2)
        plt.scatter(section['darling_medians'], section['rodney_medians'], label=section['section_code'])
    
    
    rodney_medians = np.array(all_rodney)
    darling_medians = np.array(all_darling)
    slope, inter, r, _, _ = linregress(darling_medians, rodney_medians)

    plt.plot(np.linspace(0, max, 10), np.linspace(0, max, 10), c='blue', linewidth=0.5, label='1:1')
    plt.plot(darling_medians, darling_medians*slope+inter, c='black', label='Trendline')
    plt.title(rf'$R^2$: {r**2:.4f}, Slope: {slope:.2f}')
    plt.xlabel(r'Darling Stalk Stiffness (N/$m^2$)')
    plt.ylabel(r'Rodney Stalk Stiffness (N/$m^2$)')
    plt.legend()

    # plt.figure()
    # plt.scatter(all_darling, all_rodney, label=f'All {date} tests')
    # plt.plot(np.linspace(0, max, 10), np.linspace(0, max, 10), c='blue', linewidth=0.5, label='1:1')
    # plt.legend()

# Interactive processing
def display_and_clip_tests(dates, test_nums, show_accels=False, num_stalks=0):
    for date in dates:
        for test_num in test_nums:
            test = FieldStalkSection(date=date, test_num=test_num)
            if test.exist:
                print(test_num)
                test.smooth_raw_data()
                test.shift_initials(time_cutoff=1.0)
                test.calc_force_position()
                test.differentiate_force_position()
                test.differentiate_force_position_DT()
                # test.calc_angles()
                # test.plot_force_position(view_stalks=True, show_accels=show_accels)
                test.interactive_clip_and_save(num_stalks)
    plt.show()

def interactive_process_clipped_stalks(dates, select_spans=True):
    for date in dates:
        folder = f'Results/Field/{date}'
        if not os.path.exists(folder):
            continue
        stalk_types = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        for stalk_type in stalk_types:
            response = input(f"Process stalk type '{stalk_type}'? (y/n): ").strip().lower()
            if response != 'y':
                continue
            # Load one section for height, yaw, max_position
            for test_num in range(1, 11):
                section = FieldStalkSection(date=date, test_num=test_num)
                if section.exist:
                    break
            else:
                continue
            dummy_section = type('DummySection', (), {'height': section.height, 'yaw': section.yaw, 'max_position': section.max_position})
            
            subfolder = os.path.join(folder, os.path.join(stalk_type, 'Stalk Clips'))
            csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv')]
            
            stalk_dict = {}
            for csv_file in csv_files:
                parts = csv_file[:-4].split('_')
                stalk_num = int(parts[0][1:])
                test_num = int(parts[1])
                path = os.path.join(subfolder, csv_file)
                df = pd.read_csv(path)
                time = df['Time'].to_numpy()
                force = df['Force'].to_numpy()
                position = df['Position'].to_numpy()
                stalk = StalkInteraction(time, force, position, dummy_section)
                if stalk_num not in stalk_dict:
                    stalk_dict[stalk_num] = []
                stalk_dict[stalk_num].append((test_num, stalk))
            
            # Process each stalk and collect results
            for stalk_num, stalks in sorted(stalk_dict.items()):
                results = []
                for test_num, stalk in sorted(stalks):
                    if select_spans:
                        stalk.interactive_calc_stiffness(stalk_num, test_num, date, stalk_type)
                    else:
                        stalk.recalc_stiffness(stalk_num, test_num, date, stalk_type)
                    results.append({
                        'Stalk': f'S{stalk_num:02d}',
                        'Test': f'Test_{test_num:02d}',
                        'stiffness': stalk.stiffness
                    })
                
                # Create DataFrame and pivot to have tests as columns
                results_df = pd.DataFrame(results)
                pivoted_df = results_df.pivot(index='Stalk', columns='Test', values='stiffness').reset_index()

                # Calculate average and standard deviation across test columns
                test_columns = [col for col in pivoted_df.columns if col.startswith('Test_')]
                pivoted_df['Mean'] = pivoted_df[test_columns].mean(axis=1)
                pivoted_df['Median'] = pivoted_df[test_columns].median(axis=1)
                pivoted_df['Std_Dev'] = pivoted_df[test_columns].std(axis=1)
                
                # Save to CSV in Stiffnesses subfolder
                subfolder = os.path.join(folder, stalk_type)
                os.makedirs(subfolder, exist_ok=True)
                results_path = os.path.join(subfolder, f'stiffness_{date}_{stalk_type}.csv')
                
                # Append to CSV (mode='a' adds to bottom, header only if file is new)
                pivoted_df.to_csv(results_path, mode='a', index=False, header=not os.path.exists(results_path))

def show_section_results_interactive(dates, stalk_types, correlation_flag=False):
    # Setup section on a date and verify all files are present
    sect_res = TestResults()
    for date in dates:
        parent_folder = rf'Results\Field\{date}'
        if not os.path.exists(parent_folder):
            print(f'No results for date {date}')
            continue
        
        for stalk_type in stalk_types:
            subfolder = os.path.join(parent_folder, stalk_type)
            if not os.path.exists(subfolder):
                print(f'No results for type {stalk_type} on date {date}')
                continue
            if not (all(os.path.exists(os.path.join(subfolder, x)) for x in ['Stalk Clips', 'Stalk Traces']) and 
                any(f.startswith('darling') for f in os.listdir(subfolder)) and 
                any(f.startswith('stiffness') for f in os.listdir(subfolder))):
                print(f'Required folders or files missing in subfolder {subfolder}')
                continue

            sect_res.show_results_interactive(subfolder, correlation_flag=True)

def show_day_results_interactive(dates, stalk_types, n=0):
    # Setup section on a date and verify all files are present
    boxes = []
    for date in dates:
        parent_folder = rf'Results\Field\{date}'
        if not os.path.exists(parent_folder):
            print(f'No results for date {date}')
            continue

        all_d_medians = []; all_r_medians = []; d_list = []; r_list = []
        for i, stalk_type in enumerate(stalk_types):
            subfolder = os.path.join(parent_folder, stalk_type)
            if not os.path.exists(subfolder):
                print(f'No results for type {stalk_type} on date {date}')
                continue
            if not (all(os.path.exists(os.path.join(subfolder, x)) for x in ['Stalk Clips', 'Stalk Traces']) and 
                any(f.startswith('darling') for f in os.listdir(subfolder)) and 
                any(f.startswith('stiffness') for f in os.listdir(subfolder))):
                print(f'Required folders or files missing in subfolder {subfolder}')
                continue

            d_path = os.path.join(subfolder, f'darling_{date}_{stalk_type}.csv')
            r_path = os.path.join(subfolder, f'stiffness_{date}_{stalk_type}.csv')
            
            # Read the CSV files
            darling_df = pd.read_csv(d_path, index_col=0); rodney_df = pd.read_csv(r_path, index_col=0)
            stalks_results = [rodney_df.loc[stalk][:-3].dropna().to_numpy() for stalk in rodney_df.index]
            stalks_results = stalks_results
            # print(stalks_results)
            d_means = darling_df['Mean']; d_medians = darling_df['Median']; d_stds = darling_df['Std_Dev']
            r_means = rodney_df['Mean']; r_medians = rodney_df['Median']; r_stds = rodney_df['Std_Dev']
            all_d_medians.extend(d_medians); all_r_medians.extend(r_medians)
            # boxes.append(r_medians)

            plt.figure(300+n)
            plt.scatter(d_medians, r_medians, label=f'{stalk_type}')
            section_vals = []
            for i in range(len(stalks_results)):
                section_vals.extend(stalks_results[i])
                d_list.extend([d_medians[i]]*len(stalks_results[i])); r_list.extend(stalks_results[i])
            boxes.append(section_vals)

        # plt.figure(300+n)
        # plt.scatter(d_list, r_list, s=10)

        slope, inter, r, _, _ = linregress(all_d_medians, all_r_medians)
        # slope_a, inter_a, r_a, _, _ = linregress(d_list, r_list)
        plt.figure(300+n)
        plt.plot(all_d_medians, all_d_medians, c='black', linewidth=0.5)
        plt.plot(all_d_medians, slope*np.array(all_d_medians) + inter, c='orange', linewidth=0.5)
        # plt.plot(d_list, slope_a*np.array(d_list) + inter_a, c='purple', linewidth=0.5)
        plt.title(f'Date: {date}\n'+ rf'$R^2$: {r**2:.4f}, Slope: {slope:.3f}')# | $R^2$: {r_a**2:.4f}, Slope: {slope_a:.3f}')
        plt.xlabel(r'Darling Stiffness (N/$m^2$)'); plt.ylabel(r'Hi-STIFS Stiffness (N/$m^2$)')
        plt.axis('equal')
        plt.legend()

    plt.figure(200+n)
    labels = ['Vigor (cut) 15\u00B0', 'Vigor 15\u00B0', 'Ornamental 15\u00B0', 'Xtra Early 15\u00B0', 
              'Popcorn 15\u00B0', 'Ornamental 20\u00B0', 'Vigor 20\u00B0', 'Popcorn 20\u00B0', 'Xtra Early 20\u00B0']
    starts, news = [6, 6, 8], [2, 4, 6]
    for start, new in zip(starts, news):
        boxes.insert(new, boxes.pop(start)); labels.insert(new, labels.pop(start))
    box = plt.boxplot(boxes, positions=range(len(boxes)), tick_labels=labels, patch_artist=True, notch=True)
    colors = ['red']*3 + ['green']*2 + ['blue']*2 + ['orange']*2
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel(r'Flexural Stiffness (N/$m^2$)', fontsize=20)
    plt.tick_params(axis='x', labelsize=14)

    # plt.figure(250+n)
    # new_boxes = [boxes[0]+boxes[1]+boxes[2]] + [boxes[3]+boxes[4]] + [boxes[5]+boxes[6]] + [boxes[7]+boxes[8]]
    # new_labels = ['All Vigor', 'All Ornamental', 'All X-tra Early', 'All Popcorn']
    # box = plt.boxplot(new_boxes, positions=range(len(new_boxes)), tick_labels=new_labels, patch_artist=True)
    # new_colors = ['red'] + ['green'] + ['blue'] + ['orange']
    # for patch, color in zip(box['boxes'], new_colors):
    #     patch.set_facecolor(color)
    # plt.ylabel(r'Flexural Stiffness (N/$m^2$)', fontsize=20)
    # plt.tick_params(axis='x', labelsize=14)

    # for variety, label in zip(new_boxes, new_labels):
    #     print(f'{label} - Mean: {np.mean(variety):.2f}, StdDev: {np.std(variety):.2f} CV: {np.std(variety)/np.mean(variety):.2f}')


if __name__ == '__main__':
    # show_force_position(dates=['08_19'], test_nums=range(1, 10+1), show_accels=False)
    # display_and_clip_tests(dates=['08_19'], test_nums=range(1, 10+1), num_stalks=9)
    # interactive_process_clipped_stalks(dates=['08_22'], select_spans=False)
    # show_section_results_interactive(dates=['08_07'], stalk_types=['15-A WE'])
    # show_day_results_interactive(dates=['08_07'], stalk_types=['11-B WE', '12-C WE', '13-B WE'])
    # show_day_results_interactive(dates=['08_07'], stalk_types=['11-B WE', '12-C WE', '13-B WE', '15-A WE'], n=1)

    # show_day_results_interactive(dates=['08_22'], stalk_types=['7-A Iso', '6-A Iso', '10-A Iso', '8-C Iso', '7-B Iso', '10-A', '8-C', '7-B'])
    # show_day_results_interactive(dates=['08_22'], stalk_types=['7-A Iso', '7-B Iso', '6-A Iso', '10-A Iso', '8-C Iso'], n=1)
    # show_day_results_interactive(dates=['08_22'], stalk_types=['10-A', '8-C', '7-B'], n=2)
    # show_day_results_interactive(dates=['08_22'], stalk_types=['7-A Iso'], n=1)

    # show_day_results_interactive(dates=['08_19'], stalk_types=['med'])
    show_day_results_interactive(dates=['08_22', '08_07'], stalk_types=['7-A Iso', '7-B Iso', '6-A Iso', '10-A Iso', '8-C Iso', '11-B WE', '12-C WE', '13-B WE', '15-A WE'], n=1)

    # show_accels(dates=['08_13'], test_nums=[3])
    # process_and_store_section(dates=['08_22'], test_nums=range(1, 10+1))
    # show_section_results(dates=['08_07'], test_nums=[21], correlation_flag=True)
    # show_day_results(date='08_07', correlation_flag=True)
    
    
    plt.show()

