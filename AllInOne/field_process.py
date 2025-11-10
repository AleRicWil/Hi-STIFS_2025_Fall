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

myTimeCutoff = 0.5

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
        print('yaw',self.yaw)
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
        # print(self.pos_x)

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


class StalkInteractionPair:
    def __init__(self, front, rear, section):
        self.time_f = front['Time']
        self.force_f = front['Force']
        self.position_f = front['Position']
        self.time_r = rear['Time']
        self.force_r = rear['Force']
        self.position_r = rear['Position']

        self.height = section.height
        self.B = section.max_position
        self.yaw = section.yaw   
        self.sensor_deflect = 0.07 # meters

        self.alpha = np.radians(10)
        self.def_f =  0.0
        self.def_r =  self.sensor_deflect/np.cos(self.alpha)
        
        self.stiffness = None

    def interactive_calc_stiffness(self, stalk_num, test_num, date, stalk_type):
        from matplotlib.gridspec import GridSpec
        # Discard all position = 0
        mask_f = self.position_f != 0
        self.time_f = self.time_f[mask_f]
        self.force_f = self.force_f[mask_f]
        self.position_f = self.position_f[mask_f]

        mask_r = self.position_r != 0
        self.time_r = self.time_r[mask_r]
        self.force_r = self.force_r[mask_r]
        self.position_r = self.position_r[mask_r]

        # Filter front data to keep indices where position_f is non-increasing
        if len(self.position_f) >= 1:
            keep_indices_f = [np.argmax(self.position_f)]  # Start with the max index
            for i in range(keep_indices_f[0], len(self.position_f)):
                if self.position_f[i] <= self.position_f[keep_indices_f[-1]]:
                    keep_indices_f.append(i)

            # Check if enough points remain
            if len(keep_indices_f) < 10:
                print('Not enough points on front sensor. Setting force=0 at pos=0')
                self.force_f = np.zeros(10)
                self.position_f = np.zeros(10)
                self.time_f = np.linspace(0, 1, 10)
            else:
                # Filter front arrays using keep_indices_f
                self.time_f = np.array([self.time_f[i] for i in keep_indices_f])
                self.force_f = np.array([self.force_f[i] for i in keep_indices_f])
                self.position_f = np.array([self.position_f[i] for i in keep_indices_f])
        else:
            print('Not enough points on front sensor. Setting force=0 at pos=0')
            self.force_f = np.zeros(10)
            self.position_f = np.zeros(10)
            self.time_f = np.linspace(0, 1, 10)

        # Filter rear data to keep indices where position_r is non-increasing
        if len(self.position_r) >= 1:
            keep_indices_r = [np.argmax(self.position_r)]  # Start with the first index
            for i in range(keep_indices_r[0], len(self.position_r)):
                if self.position_r[i] <= self.position_r[keep_indices_r[-1]]:
                    keep_indices_r.append(i)

            # Check if enough points remain
            if len(keep_indices_r) < 10:
                self.stiffness = np.nan
                print('Not enough points on rear sensor')
                return
            else:
                # Filter rear arrays using keep_indices_r
                self.time_r = np.array([self.time_r[i] for i in keep_indices_r])
                self.force_r = np.array([self.force_r[i] for i in keep_indices_r])
                self.position_r = np.array([self.position_r[i] for i in keep_indices_r])

        # Create figure with GridSpec for two halves
        fig = plt.figure(figsize=(12, 9))  # Adjusted width for two halves
        gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

        # Left half (front, 'f') - Top row
        ax_f_force = fig.add_subplot(gs[0, 0])  # Spans first two columns
        ax_f_position = fig.add_subplot(gs[0, 1], sharex=ax_f_force)  # Spans last two columns
        # Left half - Bottom row
        ax_f_force_pos = fig.add_subplot(gs[1, 0:2])  # Full width of left half

        # Right half (rear, 'r') - Top row
        ax_r_force = fig.add_subplot(gs[0, 2], sharex=None)  # Spans last two columns
        ax_r_position = fig.add_subplot(gs[0, 3], sharex=ax_r_force)  # Spans beyond (adjusted below)
        # Right half - Bottom row
        ax_r_force_pos = fig.add_subplot(gs[1, 2:4])  # Full width of right half

        # Adjust GridSpec to ensure proper column allocation
        gs.update(wspace=0.4, hspace=0.4)  # Spacing between subplots

        # Set figure window position
        fig.canvas.manager.window.setGeometry(100, 100, 1200, 900)  # Adjusted for wider figure

        # Gradient color based on time - Front
        indices_f = np.arange(len(self.time_f))
        norm_f = plt.Normalize(indices_f.min(), indices_f.max())
        cmap = plt.get_cmap('viridis')

        # Gradient color based on time - Rear
        indices_r = np.arange(len(self.time_r))
        norm_r = plt.Normalize(indices_r.min(), indices_r.max())

        # Front plots
        ax_f_force.scatter(self.time_f, self.force_f, s=5, c=indices_f, cmap=cmap, norm=norm_f)
        ax_f_force.set_ylabel('Force (N)')
        ax_f_force.set_xlabel('Time (s)')
        ax_f_force.set_title('Front: Force vs Time')

        ax_f_position.scatter(self.time_f, self.position_f, s=5, c=indices_f, cmap=cmap, norm=norm_f)
        ax_f_position.set_ylabel('Position (m)')
        ax_f_position.set_xlabel('Time (s)')
        ax_f_position.set_title('Front: Position vs Time')

        ax_f_force_pos.scatter(self.position_f, self.force_f, s=5, c=indices_f, cmap=cmap, norm=norm_f)
        ax_f_force_pos.set_ylabel('Force (N)')
        ax_f_force_pos.set_xlabel('Position (m)')
        ax_f_force_pos.set_title('Front: Force vs Position')

        # Rear plots
        ax_r_force.scatter(self.time_r, self.force_r, s=5, c=indices_r, cmap=cmap, norm=norm_r)
        ax_r_force.set_ylabel('Force (N)')
        ax_r_force.set_xlabel('Time (s)')
        ax_r_force.set_title('Rear: Force vs Time')

        ax_r_position.scatter(self.time_r, self.position_r, s=5, c=indices_r, cmap=cmap, norm=norm_r)
        ax_r_position.set_ylabel('Position (m)')
        ax_r_position.set_xlabel('Time (s)')
        ax_r_position.set_title('Rear: Position vs Time')

        ax_r_force_pos.scatter(self.position_r, self.force_r, s=5, c=indices_r, cmap=cmap, norm=norm_r)
        ax_r_force_pos.set_ylabel('Force (N)')
        ax_r_force_pos.set_xlabel('Position (m)')
        ax_r_force_pos.set_title('Rear: Force vs Position')

        # Overall figure title
        plt.suptitle(f'Select spans on FvX for stiffness calculation\nStalk: {stalk_num}, Test: {test_num}')

        # Select front and rear spans
        self.selected_spans_f = []
        self.current_span_f = None
        self.current_span_patches_f = {'force_pos': None, 'force_time': None, 'pos_time': None}
        self.selected_spans_r = []
        self.current_span_r = None
        self.current_span_patches_r = {'force_pos': None, 'force_time': None, 'pos_time': None}

        def onselect_f(xmin, xmax):
            # Clear previous spans
            for patch in self.current_span_patches_f.values():
                if patch:
                    patch.remove()

            self.current_span_f = (xmin, xmax)
            # Convert position range to time range for force vs time and position vs time plots
            time_mask = (self.position_f >= xmin) & (self.position_f <= xmax)
            if np.any(time_mask):
                time_min, time_max = np.min(self.time_f[time_mask]), np.max(self.time_f[time_mask])
                # Draw spans on all plots
                self.current_span_patches_f['force_time'] = ax_f_force.axvspan(time_min, time_max, alpha=0.5, facecolor='tab:blue')
                self.current_span_patches_f['pos_time'] = ax_f_position.axvspan(time_min, time_max, alpha=0.5, facecolor='tab:blue')
                self.current_span_patches_f['force_pos'] = ax_f_force_pos.axvspan(xmin, xmax, alpha=0.5, facecolor='tab:blue')
            fig.canvas.draw_idle()

        def onselect_r(xmin, xmax):
            # Clear previous spans
            for patch in self.current_span_patches_r.values():
                if patch:
                    patch.remove()

            self.current_span_r = (xmin, xmax)
            # Convert position range to time range for force vs time and position vs time plots
            time_mask = (self.position_r >= xmin) & (self.position_r <= xmax)
            if np.any(time_mask):
                time_min, time_max = np.min(self.time_r[time_mask]), np.max(self.time_r[time_mask])
                # Draw spans on all plots
                self.current_span_patches_r['force_time'] = ax_r_force.axvspan(time_min, time_max, alpha=0.5, facecolor='tab:blue')
                self.current_span_patches_r['pos_time'] = ax_r_position.axvspan(time_min, time_max, alpha=0.5, facecolor='tab:blue')
                self.current_span_patches_r['force_pos'] = ax_r_force_pos.axvspan(xmin, xmax, alpha=0.5, facecolor='tab:blue')
            fig.canvas.draw_idle()

        span_f = SpanSelector(ax_f_force_pos, onselect_f, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='tab:blue'))
        span_r = SpanSelector(ax_r_force_pos, onselect_r, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='tab:blue'))

        def add(event):
            if self.current_span_f and self.current_span_r:
                self.selected_spans_f.append(self.current_span_f)
                self.selected_spans_r.append(self.current_span_r)

                # Convert position range to time range for green spans
                time_mask = (self.position_f >= self.current_span_f[0]) & (self.position_f <= self.current_span_f[1])
                if np.any(time_mask):
                    time_min, time_max = np.min(self.time_f[time_mask]), np.max(self.time_f[time_mask])
                    # Draw green spans on all plots
                    ax_f_force.axvspan(time_min, time_max, alpha=0.3, facecolor='green')
                    ax_f_position.axvspan(time_min, time_max, alpha=0.3, facecolor='green')
                    ax_f_force_pos.axvspan(self.current_span_f[0], self.current_span_f[1], alpha=0.3, facecolor='green')

                # Convert position range to time range for green spans
                time_mask = (self.position_r >= self.current_span_r[0]) & (self.position_r <= self.current_span_r[1])
                if np.any(time_mask):
                    time_min, time_max = np.min(self.time_r[time_mask]), np.max(self.time_r[time_mask])
                    # Draw green spans on all plots
                    ax_r_force.axvspan(time_min, time_max, alpha=0.3, facecolor='green')
                    ax_r_position.axvspan(time_min, time_max, alpha=0.3, facecolor='green')
                    ax_r_force_pos.axvspan(self.current_span_r[0], self.current_span_r[1], alpha=0.3, facecolor='green')

                # Clear current span
                for patch in self.current_span_patches_f.values():
                    if patch:
                        patch.remove()
                for patch in self.current_span_patches_r.values():
                    if patch:
                        patch.remove()

                self.current_span_f = None
                self.current_span_patches_f = {'force_pos': None, 'force_time': None, 'pos_time': None}
                self.current_span_r = None
                self.current_span_patches_r = {'force_pos': None, 'force_time': None, 'pos_time': None}
                fig.canvas.draw_idle()
            else:
                print('Select span for FRONT and REAR')

        def done(event):
            if self.selected_spans_f and self.selected_spans_r:
                mask = np.full(len(self.position_f), False)
                for xmin, xmax in self.selected_spans_f:
                    current_mask = (self.position_f >= xmin) & (self.position_f <= xmax)
                    mask = np.logical_or(mask, current_mask)
                selected_force_f = self.force_f[mask]
                selected_pos_f = self.position_f[mask]
                selected_time_f = self.time_f[mask]
                front = {'Time': selected_time_f, 'Force': selected_force_f, 'Position': selected_pos_f}

                mask = np.full(len(self.position_r), False)
                for xmin, xmax in self.selected_spans_r:
                    current_mask = (self.position_r >= xmin) & (self.position_r <= xmax)
                    mask = np.logical_or(mask, current_mask)
                selected_force_r = self.force_r[mask]
                selected_pos_r = self.position_r[mask]
                selected_time_r = self.time_r[mask]
                rear = {'Time': selected_time_r, 'Force': selected_force_r, 'Position': selected_pos_r}

                save_stalk(front, rear)
                if len(selected_time_f) >= 2 and len(selected_time_r) >= 2:
                    avg_force_f = np.median(selected_force_f)
                    avg_force_r = np.median(selected_force_r)
                    
                    self.stiffness = self.height**3 * (avg_force_r - avg_force_f) / (self.def_r - self.def_f) / 3
                else:
                    self.stiffness = np.nan
            else:
                self.stiffness = np.nan
            plt.close(fig)

        def reject(event):
            self.stiffness = np.nan
            plt.close(fig)

        def save_stalk(front, rear):
            os.makedirs(f'Results/Field/{date}/{stalk_type}/Stalk Traces', exist_ok=True)
            
            df_f = pd.DataFrame({'Time': front['Time'], 
                               'Force': front['Force'], 
                               'Position': front['Position'],
                               'Sensor': 'f'})
            df_r = pd.DataFrame({'Time': rear['Time'], 
                               'Force': rear['Force'], 
                               'Position': rear['Position'],
                               'Sensor': 'r'})
            df = pd.concat([df_f, df_r]).sort_values('Time').reset_index(drop=True)
    
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
                if row[0] == "configuration":
                    self.configuration = row[1]
                    params_read += 1
                    parts = self.configuration.split()
                    self.two_sensor_flag = False
                    if 'Two' in parts and 'Straights' in parts:
                        self.two_sensor_flag = True
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
        self.strain_B1_ini = np.mean(self.strain_B1[0:cutoff_index])
        self.strain_B2_ini = np.mean(self.strain_B2[0:cutoff_index]) 
        self.c_1 = self.strain_1_ini
        self.c_2 = self.strain_2_ini
        self.c_B1 = self.strain_B1_ini
        self.c_B2 = self.strain_B2_ini

    def calc_force_position(self, smooth=True, window=20, order=1, small_den_cutoff=5*1e-5):
        self.force_num = self.k_2*(self.strain_1 - self.c_1) - self.k_1*(self.strain_2 - self.c_2)
        self.force_den = self.k_1*self.k_2*(self.d_2 - self.d_1)
        self.force = np.where(self.force_num / self.force_den < 0, 0, self.force_num / self.force_den)
        self.force_raw = self.force_num / self.force_den

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

    def plot_force_position(self, view_stalks=False, plain=True, show_accels=False):
        try:
            time_ini = self.stalks[0].time_loc
            time_end = self.stalks[-1].time_loc
        except:
            time_ini = 0
            time_sep = 4.47
            time_end = self.time[-1]
        if show_accels:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9.5, 4.8))
        else:
            if self.two_sensor_flag:
                fig, ax = plt.subplots(4,1, sharex=True, figsize=(15,10))
                ax[0].plot(self.time - time_sep, self.force, label='Force')
                # ax[0].plot(self.time - time_ini, self.force_raw, label='raw', linewidth=0.5)
                ax[0].set_ylabel('Force Rear (N)')

                ax[1].plot(self.time, self.forceB, label='Force_B')
                ax[1].set_ylabel('Force Front (N)')
                
                ax[2].plot(self.time - time_sep, self.position*100, label='Position')
                # ax[2].plot(self.time - time_ini, self.position_raw*100, label='raw', linewidth=0.5)
                ax[2].set_ylabel('Position Rear (cm)')
                
                ax[3].plot(self.time, self.positionB*100, label='Position_B')
                ax[3].set_ylabel('Position Front (cm)')
                ax[3].set_xlabel('Time (s)')
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
        fig.canvas.manager.window.move(10, 10)

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
            raise ValueError('No interactions to collect stalks')
        
        
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

    def two_interactive_clip_and_save(self, num_stalks):
        # Display force and position plots for user selection
        fig, ax = plt.subplots(4,1, sharex=True, figsize=(16,10))
        ax[0].plot(self.time, self.force, label='Force')
        ax[0].set_ylabel('Force Rear (N)')

        ax[1].plot(self.time, self.forceB, label='Force_B')
        ax[1].set_ylabel('Force Front (N)')
        
        ax[2].plot(self.time, self.position*100, label='Position')
        ax[2].set_ylabel('Position Rear (cm)')
        
        ax[3].plot(self.time, self.positionB*100, label='Position_B')
        ax[3].set_ylabel('Position Front (cm)')
        ax[3].set_xlabel('Time (s)')
        plt.suptitle(f'{self.date} Test {self.test_num} {self.stalk_type}\nSelect {num_stalks} stalk interactions')
        fig.canvas.manager.window.move(10, 10)

        # Allow user to select spans from plotted data, from front and rear sensors
        self.spans_f = []
        self.spans_r = []
        self.current_span_f = None
        self.current_span_r = None
        self.current_span_patch_f = None
        self.current_span_patch_r = None
        self.current_span_patch_f_force = None
        self.current_span_patch_r_force = None

        def onselect_f(xmin, xmax):
            # Remove previous temporary front span if exists
            if self.current_span_patch_f:
                self.current_span_patch_f.remove()
                self.current_span_patch_f_force.remove()
            self.current_span_f = (xmin, xmax)
            self.current_span_patch_f = ax[3].axvspan(xmin, xmax, alpha=0.5, color='tab:blue')
            self.current_span_patch_f_force = ax[1].axvspan(xmin, xmax, alpha=0.5, color='tab:blue')
            fig.canvas.draw_idle()
            print('FRONT', self.current_span_f)

        def onselect_r(xmin, xmax):
            # Remove previous temporary rear span if exists
            if self.current_span_patch_r:
                self.current_span_patch_r.remove()
                self.current_span_patch_r_force.remove()
            self.current_span_r = (xmin, xmax)
            self.current_span_patch_r = ax[2].axvspan(xmin, xmax, alpha=0.5, color='tab:blue')
            self.current_span_patch_r_force = ax[0].axvspan(xmin, xmax, alpha=0.5, color='tab:blue')
            fig.canvas.draw_idle()
            print('REAR', self.current_span_r)

        self.span_f = SpanSelector(ax[3], onselect_f, 'horizontal', useblit=True, 
                                   props=dict(alpha=0.5, facecolor='tab:blue'))
        self.span_r = SpanSelector(ax[2], onselect_r, 'horizontal', useblit=True, 
                                   props=dict(alpha=0.5, facecolor='tab:blue'))
        
        # Create UI elements to save spans
        def confirm(event):
            if self.current_span_f and self.current_span_r:
                print(f'Confirming spans: FRONT: {self.current_span_f}, REAR: {self.current_span_r}')
                self.spans_f.append(self.current_span_f)
                self.spans_r.append(self.current_span_r)
                if self.current_span_patch_f: self.current_span_patch_f.remove(); self.current_span_patch_f_force.remove()
                if self.current_span_patch_r: self.current_span_patch_r.remove(); self.current_span_patch_r_force.remove()
                ax[3].axvspan(self.current_span_f[0], self.current_span_f[1], alpha=0.3, color='green')
                ax[1].axvspan(self.current_span_f[0], self.current_span_f[1], alpha=0.3, color='red')
                ax[2].axvspan(self.current_span_r[0], self.current_span_r[1], alpha=0.3, color='green')
                ax[0].axvspan(self.current_span_r[0], self.current_span_r[1], alpha=0.3, color='red')
                self.current_span_f = None
                self.current_span_r = None
                self.current_span_patch_f = None
                self.current_span_patch_r = None
                fig.canvas.draw_idle()
                if len(self.spans_f) == num_stalks and len(self.spans_r) == num_stalks:
                    save_stalks()
                    plt.close(fig)
            else: print('Front and Rear spans need to be selected to confirm')
        #
            # trigger confirm with button
        ax_confirm = plt.axes([0.8, 0.025, 0.1, 0.075])
        btn_confirm = Button(ax_confirm, 'Confirm')
        btn_confirm.on_clicked(confirm)

            # trigger confirm with 'enter' key
        def on_key(event):
            if event.key == 'enter':
                confirm(event)
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Save spans
        def save_stalks():
            os.makedirs(f'Results/Field/{self.date}/{self.stalk_type}/Stalk Clips', exist_ok=True)
            for i, ((tmin_f, tmax_f), (tmin_r, tmax_r)) in enumerate(zip(self.spans_f, self.spans_r)):
                # Clip 'f' span
                mask_f = (self.time >= tmin_f) & (self.time <= tmax_f)
                df_f = pd.DataFrame({
                    'Time': self.time[mask_f],
                    'Force': self.forceB[mask_f],
                    'Position': self.positionB[mask_f],
                    'Sensor': 'f'  # Metadata for easy filtering later
                })

                # Clip 'r' span
                mask_r = (self.time >= tmin_r) & (self.time <= tmax_r)
                df_r = pd.DataFrame({
                    'Time': self.time[mask_r],
                    'Force': self.force[mask_r],
                    'Position': self.position[mask_r],
                    'Sensor': 'r'  # Metadata for easy filtering later
                })

                # Combine and sort by time (optional but recommended for chronological consistency)
                df = pd.concat([df_f, df_r]).sort_values('Time').reset_index(drop=True)

                # Save to CSV
                path = f'Results/Field/{self.date}/{self.stalk_type}/Stalk Clips/S{i+1:02d}_{self.test_num:02d}.csv'
                df.to_csv(path, index=False)
                print(f"Saved stalk {i+1} (both spans) to {path}")  # Updated debug output

        plt.show()

    def plot_raw_StrainForce(self):
        fig, ax = plt.subplots(3,1, sharex=True, figsize=(9,7))

        ax[0].plot(self.time, self.strain_1_raw, linewidth=0.5)
        ax[0].axhline(np.percentile(self.strain_1_raw, 99), c='red')
        ax[0].axhline(np.percentile(self.strain_1_raw, 1), c='red')
        ax[0].set_ylabel('Strain 1', fontsize=14)

        ax[1].plot(self.time, self.strain_2_raw, linewidth=0.5)
        ax[1].axhline(np.percentile(self.strain_2_raw, 99), c='red')
        ax[1].axhline(np.percentile(self.strain_2_raw, 1), c='red')
        ax[1].set_ylabel('Strain 2', fontsize=14)

        ax[2].plot(self.time, self.force_raw, linewidth=0.5)
        ax[2].axhline(np.percentile(self.force_raw, 99), c='red')
        ax[2].axhline(np.percentile(self.force_raw, 1), c='red')
        ax[2].set_ylabel('Force (N)', fontsize=14)
        
        ax[-1].set_xlabel('Time (s)')

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
        # d_path = os.path.join(filepath, f'darling_{date}_{stalk_type}.csv')
        h_path = os.path.join(filepath, f'stiffness_{date}_{stalk_type}.csv')
        
        # Read the CSV files
        # darling_df = pd.read_csv(d_path, index_col=0)
        HiSTIFFS_df = pd.read_csv(h_path, index_col=0)
        stalks_results = [HiSTIFFS_df.loc[stalk][:-3].dropna().to_numpy() for stalk in HiSTIFFS_df.index]
        
        plt.figure()
        for i, stalk in enumerate(stalks_results):
            plt.boxplot(stalk, positions=[i+1], notch=True)

        # if correlation_flag:
        #     d_means = darling_df['Mean']; d_medians = darling_df['Median']; d_stds = darling_df['Std_Dev']
        #     r_means = rodney_df['Mean']; r_medians = rodney_df['Median']; r_stds = rodney_df['Std_Dev']
            
        #     slope, inter, r, _, _ = linregress(d_medians, r_medians)
            
        #     plt.figure()
        #     plt.scatter(d_medians, r_medians)
        #     plt.plot(d_medians, d_medians, c='black', linewidth=0.5)
        #     plt.plot(d_medians, slope*np.array(d_medians) + inter, c='orange', linewidth='0.5')
        #     plt.title(f'Date: {date}, Section: {stalk_type}\n'+ rf'$R^2$: {r**2:.4f}, Slope: {slope:.3f}')
        #     plt.axis('equal')
    

# Automatic processing
def show_force_position(dates, test_nums, show_accels):
    for date in dates:
        for test_num in test_nums:
            test = FieldStalkSection(date=date, test_num=test_num)
            if test.exist:
                test.smooth_raw_data()
                test.shift_initials(time_cutoff=myTimeCutoff)
                test.calc_force_position()
                test.differentiate_force_position()
                test.differentiate_force_position_DT()
                # test.find_stalk_interaction()
                # test.collect_stalks()
                # test.calc_section_stiffnesses()
                # test.calc_angles()

                test.plot_force_position(view_stalks=False, show_accels=show_accels)
                # test.plot_section_stiffnesses()
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
                test.shift_initials(time_cutoff=myTimeCutoff)
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
                test.shift_initials(time_cutoff=myTimeCutoff)
                test.calc_force_position()
                test.differentiate_force_position()
                test.differentiate_force_position_DT()
                # test.calc_angles()
                # test.plot_force_position(view_stalks=True, show_accels=show_accels)
                if not test.two_sensor_flag: test.interactive_clip_and_save(num_stalks)
                else: test.two_interactive_clip_and_save(num_stalks)

            
    plt.show()

def interactive_process_clipped_stalks(dates, types_to_process=None, select_spans=True):
    for date in dates:
        folder = f'Results/Field/{date}'
        if not os.path.exists(folder):
            continue
        
        stalk_types = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        for stalk_type in stalk_types:
            if types_to_process is None:
                response = input(f"Process stalk type '{stalk_type}'? (y/n): ").strip().lower()
                if response != 'y':
                    continue
            elif stalk_type not in types_to_process:
                print(f'Skipped stalks from {stalk_type}')
                continue
        
            print(f'Processing from {stalk_type}')
            # Load one section for height, yaw, max_position
            subfolder = os.path.join(folder, os.path.join(stalk_type, 'Stalk Clips'))
            csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv')]
            parts = csv_files[0][:-4].split('_')
            test_num = int(parts[1])
            for test_num in range(test_num, test_num+10):
                section = FieldStalkSection(date=date, test_num=test_num)
                if section.exist:
                    break
            else:
                continue
            
            stalk_dict = {}
            for csv_file in csv_files:
                parts = csv_file[:-4].split('_')
                stalk_num = int(parts[0][1:])
                test_num = int(parts[1])
                path = os.path.join(subfolder, csv_file)
                df = pd.read_csv(path)

                if not section.two_sensor_flag:
                    time = df['Time'].to_numpy()
                    force = df['Force'].to_numpy()
                    position = df['Position'].to_numpy()
                    stalk = StalkInteraction(time, force, position, section)
                else:
                    front = {'Time': df['Time'][df['Sensor'] == 'f'].to_numpy(),
                             'Force': df['Force'][df['Sensor'] == 'f'].to_numpy(),
                             'Position': df['Position'][df['Sensor'] == 'f'].to_numpy()}
                    
                    rear = {'Time': df['Time'][df['Sensor'] == 'r'].to_numpy(),
                             'Force': df['Force'][df['Sensor'] == 'r'].to_numpy(),
                             'Position': df['Position'][df['Sensor'] == 'r'].to_numpy()}

                    stalk = StalkInteractionPair(front, rear, section)

                if stalk_num not in stalk_dict:
                    stalk_dict[stalk_num] = []
                stalk_dict[stalk_num].append((test_num, stalk))
            
            # Process each stalk and collect results
            for stalk_num, stalks in sorted(stalk_dict.items()):
                results = []
                for test_num, stalk in sorted(stalks):
                    if select_spans:
                        # if test_num > 41:
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

def noise_character(date, test_num):
    test = FieldStalkSection(date, test_num)
    test.shift_initials(time_cutoff=myTimeCutoff)
    test.calc_force_position(smooth=False)
    test.plot_raw_StrainForce()

    per1_1, med_1, per99_1 = np.percentile(test.strain_1_raw, (1, 50, 99))
    per1_2, med_2, per99_2 = np.percentile(test.strain_2_raw, (1, 50, 99))
    per1_f, med_f, per99_f = np.percentile(test.force_raw, (1, 50, 99))

    print(f'Strain 1:\n1-99 Band: {abs(per1_1-per99_1):.6f}\n1st: {per1_1:.6f} Median: {med_1:.6f} 99th: {per99_1:.6f}')
    print(f'Strain 2:\n1-99 Band: {abs(per1_2-per99_2):.6f}\n1st: {per1_2:.6f} Median: {med_2:.6f} 99th: {per99_2:.6f}')
    print(f'Force:\n1-99 Band: {abs(per1_f-per99_f):.6f}\n1st: {per1_f:.6f} Median: {med_f:.6f} 99th: {per99_f:.6f}')

def single_stalk_boxplots(dates, stalk_types, n=0):
    all_stalks_results = []
    for date in dates:
        parent_folder = rf'Results\Field\{date}'
        if not os.path.exists(parent_folder):
            print(f'No results for date {date}')
            continue

        for i, stalk_type in enumerate(stalk_types):
            subfolder = os.path.join(parent_folder, stalk_type)
            if not os.path.exists(subfolder):
                print(f'No results for type {stalk_type} on date {date}')
                continue
            if not (all(os.path.exists(os.path.join(subfolder, x)) for x in ['Stalk Clips', 'Stalk Traces']) and
                any(f.startswith('stiffness') for f in os.listdir(subfolder))):
                print(f'Required folders or files missing in subfolder {subfolder}')
                continue

            HS_path = os.path.join(subfolder, f'stiffness_{date}_{stalk_type}.csv')
            HS_df = pd.read_csv(HS_path, index_col=0)
            stalks_results = [HS_df.loc[stalk][:-3].dropna().to_numpy() for stalk in HS_df.index]
            all_stalks_results.extend(stalks_results)


    # Generate plots
    # plt.figure()
    # plt.boxplot(all_stalks_results)#, tick_labels=['']*len(all_stalks_results))
    # plt.title('Same-Stalk Measurement Spread', fontsize=16)
    # plt.ylabel(r'Stiffness ($N/m^2$)', fontsize=14)
    # plt.xlabel('Stalk Number', fontsize=14)
    # plt.ylim(0, 60)
    # plt.legend(['Device: Hi-STIFFS B'], fontsize=12, loc='upper left')
    # plt.xticks(rotation=30, ha='right')

    # Compute summary statistics for each stalk
    means = [np.mean(stalk) for stalk in all_stalks_results]
    medians = [np.median(stalk) for stalk in all_stalks_results]
    stds = [np.std(stalk, ddof=1) for stalk in all_stalks_results]  # Sample standard deviation
    # Compute mean-median differences
    deltas = [mean - med for mean, med in zip(means, medians)]
    skew_estimates = [3 * delta / sd if sd != 0 else 0 for delta, sd in zip(deltas, stds)]
    z = 0.6745  # z-score for 25th/75th percentiles
    q1_estimates = [med - z * sd for med, sd in zip(medians, stds)]
    q3_estimates = [med + z * sd for med, sd in zip(medians, stds)]
    # Adjust quartiles with k=0.5
    k = 0.5
    q1_adj = [q1 + k * delta for q1, delta in zip(q1_estimates, deltas)]
    q3_adj = [q3 + k * delta for q3, delta in zip(q3_estimates, deltas)]

    # X positions: one per stalk
    x = np.arange(1, len(all_stalks_results)+1)

    # Plot median as main line
    plt.scatter(x, medians, color='orange', marker='_', s=50, zorder=5)

    # Add estimated Q1–Q3 as vertical lines (mimics boxplot whiskers without outliers)
    for i in range(len(x)):
        plt.vlines(x[i], q1_adj[i], q3_adj[i], color='black', linewidth=6, alpha=0.2, zorder=4)
        # Optional thin caps
        plt.hlines(q1_adj[i], x[i]-0.2, x[i]+0.2, color='black', linewidth=1)
        plt.hlines(q3_adj[i], x[i]-0.2, x[i]+0.2, color='black', linewidth=1)

    # Formatting
    plt.title('Same-Stalk Measurement Summary (Estimated Quartiles)', fontsize=16)
    plt.ylabel(r'Stiffness ($N/m^2$)', fontsize=14)
    plt.xlabel('Stalk Number', fontsize=14)
    plt.ylim(0, 60)
    plt.legend(['Device: Hi-STIFFS A'], fontsize=12, loc='upper left')
    plt.xticks(x, rotation=30, ha='right')


if __name__ == '__main__':
    
    myDates = ['10_01']; myTestNums = range(1, 1+1); myTypesToProcess = ['Vigor 1 - 15deg', 'Vigor 1 - 20deg', 'Vigor 2 - 15deg']
    # noise_character(myDates[0], myTestNums[0])
    # single_stalk_boxplots(myDates, myTypesToProcess)
    # show_force_position(dates=myDates, test_nums=myTestNums, show_accels=False)
    # display_and_clip_tests(dates=myDates, test_nums=myTestNums, num_stalks=13) 
    # interactive_process_clipped_stalks(dates=myDates, types_to_process=myTypesToProcess, select_spans=True)
    # show_section_results_interactive(dates=myDates, stalk_types=myTypesToProcess)

    # show_accels(dates=['08_13'], test_nums=[3])
    # process_and_store_section(dates=['08_22'], test_nums=range(1, 10+1))
    # show_section_results(dates=['08_07'], test_nums=[21], correlation_flag=True)
    # show_day_results(date='08_07', correlation_flag=True)
    
    
    # cv_dirt = np.array([0.1651932,  0.5,        0.11421431, 0.1061719,  0.05633551, 0.10093206,
    #                     0.35470862, 0.11357931, 0.21347642, 0.30801587, 0.15951702, 0.12420839,
    #                     0.19017683, 0.35939948, 0.33701432, 0.22981235, 0.14935651, 0.40948684,
    #                     0.20189415])
    
    # pd1 = pd.read_csv(r'Results\Field\10_01\Vigor 1 - 15deg\stiffness_10_01_Vigor 1 - 15deg.csv')
    # pd2 = pd.read_csv(r'Results\Field\10_01\Vigor 2 - 15deg\stiffness_10_01_Vigor 2 - 15deg.csv')

    # stiff1 = pd1['Median'].to_numpy()
    # stdev1 = pd1['Std_Dev'].to_numpy()
    # cv1 = stdev1 / stiff1
    # stiff2 = pd2['Median'].to_numpy()
    # stdev2 = pd2['Std_Dev'].to_numpy()
    # cv2 = stdev2 / stiff2
  

    # cv_board = np.concatenate((cv1, cv2))
    # cv_board = cv_board[~np.isnan(cv_board)]

    # cv_dirt = np.clip(cv_dirt, 0, 0.5)
    # cv_board = np.clip(cv_board, 0, 0.5)

    # print(np.median(cv_dirt))
    # print(np.median(cv_board))

    

    # t, p = stats.ttest_ind(cv_dirt, cv_board, equal_var=False)
    # print(t, p)

    one = [10.005,10.495,3.467,4.567,1.362,5.880,6.670]
    two = [36.553,52.864,41.011,32.680,28.212,36.397,44.279,4.112,15.175]
    three = [33.304,22.822,25.807,20.981,23.292,10.645,20.666,13.766,17.455]
    four = [4.754,11.286,5.784,11.667,18.226,10.708,14.081,6.949,13.608,8.705,9.097,10.411,8.176]
    five = [5.010,10.901,15.349,18.251,20.778,5.363,19.174,24.775,25.949,9.611]
    means = one + two + three + four + five

    one = [9.359,10.228,3.690,4.594,1.369,5.872,6.761]
    two = [35.571,51.891,38.954,32.993,26.660,36.259,44.627,4.169,14.671]
    three = [32.897,20.909,25.383,21.198,20.793,10.685,20.524,14.113,17.377]
    four = [4.189,11.422,5.942,12.155,18.134,11.574,14.520,6.980,13.344,7.938,9.354,9.422,7.972]
    five = [6.901,10.776,14.601,18.367,21.191,5.206,19.044,25.839,26.215,9.619]
    medians = one + two + three + four + five

    one = [2.065,3.006,0.709,0.788,0.014,0.686,0.358]
    two = [5.145,5.562,8.920,4.174,4.343,4.877,2.748,0.407,2.956]
    three = [7.613,4.694,2.675,1.378,3.580,1.538,4.489,2.052,1.088]
    four = [0.973,1.399,0.575,1.376,1.683,1.787,1.968,0.261,1.282,2.464,0.893,1.980,1.312]
    five = [3.482,1.534,5.971,0.326,1.219,0.304,1.048,3.057,1.896,1.024]
    stds = one + two + three + four + five

    # Compute mean-median differences
    deltas = [mean - med for mean, med in zip(means, medians)]
    skew_estimates = [3 * delta / sd if sd != 0 else 0 for delta, sd in zip(deltas, stds)]
    z = 0.6745  # z-score for 25th/75th percentiles
    q1_estimates = [med - z * sd for med, sd in zip(medians, stds)]
    q3_estimates = [med + z * sd for med, sd in zip(medians, stds)]
    # Adjust quartiles with k=0.5
    k = 0.5
    q1_adj = [q1 + k * delta for q1, delta in zip(q1_estimates, deltas)]
    q3_adj = [q3 + k * delta for q3, delta in zip(q3_estimates, deltas)]

    # X positions: one per stalk
    x = np.arange(1, len(means)+1)

    # Plot median as main line
    plt.scatter(x, medians, color='orange', marker='_', s=50, zorder=5)

    # Add estimated Q1–Q3 as vertical lines (mimics boxplot whiskers without outliers)
    for i in range(len(x)):
        plt.vlines(x[i], q1_adj[i], q3_adj[i], color='black', linewidth=6, alpha=0.2, zorder=4)
        # Optional thin caps
        plt.hlines(q1_adj[i], x[i]-0.2, x[i]+0.2, color='black', linewidth=1)
        plt.hlines(q3_adj[i], x[i]-0.2, x[i]+0.2, color='black', linewidth=1)

    # Formatting
    plt.title('Same-Stalk Measurement Summary (Estimated Quartiles)', fontsize=16)
    plt.ylabel(r'Stiffness ($N/m^2$)', fontsize=14)
    plt.xlabel('Stalk Number', fontsize=14)
    plt.ylim(0, 60)
    plt.legend(['Device: DARLING'], fontsize=12, loc='upper left')
    plt.xticks(x, rotation=30, ha='right')



    plt.show()

