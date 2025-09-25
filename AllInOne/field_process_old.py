import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import os
from skopt import gp_minimize
from skopt.space import Real
from scipy.signal import savgol_filter
import time

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
        self.stiffness = {}
        self.height = section.height
        self.yaw = section.yaw

    def calc_stalk_stiffness(self):
        # Fit linear regression to force and position vs time
        time = self.time
        force = self.force
        position = self.position
        slope_f, intercept_f, r_f, _, _ = stats.linregress(time, force)
        slope_p, intercept_p, r_p, _, _ = stats.linregress(time, position)
        print(f'R^2: {r_f**2}, {r_p**2}')

        count = 0
        prev_len = len(time)
        while r_f**2 < 0.85 or r_p**2 < 0.85 and len(time) > 30 and count < 10:
            count += 1
            if prev_len <= len(time) and count > 1:
                ipr_scale = 1.0
            else:
                ipr_scale = 3.0
            # print(f'Starting iterative fit with {len(time)} points\nR^2: {r_f**2}, {r_p**2}')
            fit_f = np.polyval([slope_f, intercept_f], time)
            fit_p = np.polyval([slope_p, intercept_p], time)
            # plt.plot(time, force)
            # plt.plot(time, fit_f)
            # plt.ylim(0,60)
            # plt.figure()
            # plt.plot(time, position)
            # plt.plot(time, fit_p)
            # plt.ylim(0.05, 0.20)
            # plt.show()
            
            residuals_f = np.abs(force - fit_f)
            residuals_p = np.abs(position - fit_p)
            
            p05_f, p95_f = np.percentile(residuals_f, [5, 95])
            ipr_f = p95_f - p05_f
            # print(ipr_scale)
            threshold_f = ipr_scale * (1 - r_f**2) * ipr_f
            
            # p1_p, p9_p = np.percentile(residuals_p, [10, 90])
            # ipr_p = p9_p - p1_p
            # threshold_p = 1.0 * (1 - r_p**2) * ipr_p
            
            mask = (residuals_f <= threshold_f)# & (residuals_p <= threshold_p)
            
            force = force[mask]
            position = position[mask]
            prev_len = len(time)
            time = time[mask]
            slope_f, intercept_f, r_f, _, _ = stats.linregress(time, force)
            slope_p, intercept_p, r_p, _, _ = stats.linregress(time, position)

        fit_f = np.polyval([slope_f, intercept_f], time)
        fit_p = np.polyval([slope_p, intercept_p], time)
        # print(f'Ending iterative fit with {len(time)} points\nR^2: {r_f**2}, {r_p**2}')
        # plt.plot(time, force)
        # plt.plot(time, fit_f)
        # plt.ylim(0,60)
        # plt.figure()
        # plt.plot(time, position)
        # plt.plot(time, fit_p)
        # plt.ylim(0.05, 0.20)
        # plt.show()
        
        # Create line to visualize on plots
        self.fits['time'] = time
        self.fits['force'] = np.polyval([slope_f, intercept_f], self.time)
        self.fits['position'] = np.polyval([slope_p, intercept_p], self.time)

        # Calulate fluxual stiffness from slopes and system parameters
        num = slope_f*self.height**3
        den = -3*slope_p*np.sin(self.yaw) # negate because position starts at end of sensor beam (15cm) and ends at base (5cm)
        self.stiffness = num/den

class FieldStalkSection:
    def __init__(self, date, test_num, min_force_rate=-0.5, pos_accel_tol=0.8, force_accel_tol=700):
        # These params set the filter bounds for identifying which portions of the data are stalk interactions. These are ideally straight lines, increasing in
        # force and decreasing in position. 
            # this window should be a bit wider than the physical sensor
        self.min_position = 5*1e-2  # centimeters, location of 2nd strain gauge
        self.max_position = 18*1e-2 # centimeters, location of beam end
        self.min_force = 1 # newton, easily cut out the spaces between stalks. Rejects noisy detection regime
        self.min_force_rate = min_force_rate    # newton/sec, only look at data where the stalk is being pushed outward
        self.max_force_rate = 80                # newton/sec, reject stalk first falling onto sensor beam
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

    def plot_force_position(self, view_stalks=False, plain=True):
        try:
            time_ini = self.stalks[0].time_loc
            time_end = self.stalks[-1].time_loc
        except:
            time_ini = 0
            time_end = 10
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
        ax[0].plot(self.time - time_ini, self.force, label='Force')
        ax[0].set_ylabel('Force (N)')
        ax[1].plot(self.time - time_ini, self.position*100, label='Position')
        ax[0].plot(self.time - time_ini, self.force_raw, label='raw', linewidth=0.5)
        ax[1].plot(self.time - time_ini, self.position_raw*100, label='raw', linewidth=0.5)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (cm)')

        plt.suptitle(f'{self.configuration}, Date:{self.date}, Test #{self.test_num}\nStalks:{self.stalk_type}')
        plt.xlim(-2, time_end - time_ini + 2)

        if view_stalks:
            for stalk in self.stalks:
                if not np.isnan(stalk.time).all():
                    ax[0].plot(stalk.time - time_ini, stalk.force, c='red')
                    ax[1].plot(stalk.time - time_ini, stalk.position*100, c='red')
                    if hasattr(self, 'force_fits'):
                        ax[0].plot(stalk.time - time_ini, stalk.fits['force'], c='green')
                        ax[1].plot(stalk.time - time_ini, stalk.fits['position']*100, c='green')
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

    # def find_stalk_interaction(self):
    #     # Combine conditions into a single boolean mask
    #     mask = (np.abs(self.position_DDT) < self.pos_accel_tol) & \
    #         (self.position > self.min_position) & \
    #         (self.position < self.max_position) & \
    #         (self.position_DT < self.max_pos_rate) & \
    #         (self.force > self.min_force) & \
    #         (self.force_DT < self.max_force_rate) & \
    #         (self.force_DT > self.min_force_rate) & \
    #         (np.abs(self.force_DDT) < self.force_accel_tol)
        
    #     # Get initial interaction indices
    #     interaction_indices = np.where(mask)[0]

    #     # Filter out blips (groups with fewer than min_sequential indices)
    #     if len(interaction_indices) > 0:
    #         diffs = np.diff(interaction_indices)
    #         group_starts = np.where(diffs > 1)[0] + 1
    #         groups = np.split(interaction_indices, group_starts)
    #         interaction_indices = np.concatenate([g for g in groups if len(g) >= self.min_seq_points]) if groups else np.array([])
        
    #     # Reconnect gaps < 30% of average gap
    #     new_indices = []
    #     for i in range(len(interaction_indices) - 1):
    #         new_indices.append(interaction_indices[i])
    #         gap = interaction_indices[i + 1] - interaction_indices[i]
    #         if gap > 1 and gap < self.stitch_gap_limit and abs(self.force[interaction_indices[i+1]] - self.force[interaction_indices[i]]) < 2:
    #             new_indices.extend(range(interaction_indices[i] + 1, interaction_indices[i + 1]))
    #     new_indices.append(interaction_indices[-1])
    #     interaction_indices = np.array(new_indices, dtype=np.int64)

    #     # Assign results
    #     self.interaction_indices = interaction_indices
    #     self.stalk_force = self.force[interaction_indices]
    #     self.stalk_position = self.position[interaction_indices]
    #     self.stalk_time = self.time[interaction_indices]

    # def collect_stalk_sections(self):
    #     gaps = np.concatenate(([0], np.diff(self.interaction_indices)))
    #     big_gaps = gaps[gaps>1]
    #     avg_gap = np.average(big_gaps)

    #     self.stalk_forces = []
    #     self.stalk_positions = []
    #     self.stalk_times = []
    #     stalk_force_section = []
    #     stalk_position_section = []
    #     stalk_time_section = []
        
    #     for i in range(len(self.interaction_indices)):
    #         if gaps[i] <= 1:    # accumulate point on current stalk
    #             stalk_force_section.append(self.stalk_force[i])
    #             stalk_position_section.append(self.stalk_position[i])
    #             stalk_time_section.append(self.stalk_time[i])
    #         else:               # store current stalk and reset accumulation
    #             if stalk_force_section[0] < stalk_force_section[-1] and stalk_position_section[0] > stalk_position_section[-1] and \
    #                 stalk_time_section[-1] - stalk_time_section[0] >= 0.5 and \
    #                     max(stalk_position_section)*0.95 <= stalk_position_section[0]:
    #                 self.stalk_forces.append(np.array(stalk_force_section))
    #                 self.stalk_positions.append(np.array(stalk_position_section))
    #                 self.stalk_times.append(np.array(stalk_time_section))
    #             stalk_force_section = []
    #             stalk_position_section = []
    #             stalk_time_section = []
                
    #             if gaps[i] >= avg_gap*1.6:  # if the gap is very large, skip next stalk number
    #                 self.stalk_forces.append(np.nan)
    #                 self.stalk_positions.append(np.nan)
    #                 self.stalk_times.append(np.nan)
        
    #     # add the last stalk
    #     self.stalk_forces.append(np.array(stalk_force_section))
    #     self.stalk_positions.append(np.array(stalk_position_section))
    #     self.stalk_times.append(np.array(stalk_time_section))

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
        self.stalk_forces = []
        self.stalk_positions = []
        self.stalk_times = []
        
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
                self.stalk_forces.append(force)
                self.stalk_positions.append(position)
                self.stalk_times.append(time)

    def calc_section_stiffnesses(self):
        # print(f'Computing stiffness for {self.stalk_type} stalks')
        self.force_fits = []
        self.position_fits = []
        self.flex_stiffs = []
        for stalk in self.stalks:
            if not np.isnan(stalk.time).all():
                stalk.calc_stalk_stiffness()
                self.force_fits.append(stalk.fits['force'])
                self.position_fits.append(stalk.fits['position'])
                # self.stalk_times.append(stalk.time)
                self.flex_stiffs.append(stalk.stiffness)
            else:
                self.force_fits.append(np.nan)
                self.position_fits.append(np.nan)
                self.flex_stiffs.append(np.nan)
            
        results.append(self.flex_stiffs)

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

def show_force_position(dates, test_nums):
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
                test.plot_force_position(view_stalks=True)
                test.plot_section_stiffnesses()
                test.plot_section_stiffnesses()
    plt.show()


if __name__ == '__main__':
    show_force_position(dates=['08_13'], test_nums=range(1, 15+1))