import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename = 'weight test 1.csv'
filenameb = r'Raw Data\11_05\11_05_test_1.csv'
data = pd.read_csv(filename, skiprows=22)
datab = pd.read_csv(filenameb, skiprows=12)

time = data['Time']
v1 = data['Voltage_1']
v2 = data['Voltage_2']

timeb = datab['Time']
v1b = datab['Strain A1']
v2b = datab['Strain A2']

ylim = 5e-5
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(time, v1, linewidth=0.5)
ax[0].set_ylabel('Voltage 1 (V)')
ax[0].set_ylim(-ylim, ylim)
ax[1].plot(time, v2, linewidth=0.5)
ax[1].set_ylabel('Voltage 2 (V)')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylim(-ylim, ylim)

ylim = 4e-1
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(timeb, v1b, linewidth=0.5)
ax[0].set_ylabel('Strain 1 (V?)')
ax[0].set_ylim(0.2, ylim)
ax[1].plot(timeb, v2b, linewidth=0.5)
ax[1].set_ylabel('Strain 2 (V?)')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylim(0.2, ylim)



end_time = 5 # seconds
start_time = 0

time = timeb
v1 = v1b
v2 = v2b

end_ind = np.where(time >=end_time)[0][0]
start_ind = np.where(time >= start_time)[0][0]

cut_time = time[start_ind:end_ind]
cut_v1 = v1[start_ind:end_ind]
cut_v2 = v2[start_ind:end_ind]

print(f'Std: {np.std(cut_v1)}')

p1_1, p50_1, p99_1 = np.percentile(cut_v1, (1, 50, 99))
p1_2, p50_2, p99_2 = np.percentile(cut_v2, (1, 50, 99))
print(f'Voltage 1:\n1-99 Band: {abs(p1_1-p99_1):.8f}\n1st: {p1_1:.8f} Median: {p50_1:.8f} 99th: {p99_1:.8f}')
print(f'Voltage 2:\n1-99 Band: {abs(p1_2-p99_2):.8f}\n1st: {p1_2:.8f} Median: {p50_2:.8f} 99th: {p99_2:.8f}')


ylim = 1e-4
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(cut_time, cut_v1, linewidth=0.5)
ax[0].axhline(p1_1, c='red', linewidth=0.8)
ax[0].axhline(p99_1, c='red', linewidth=0.8)
ax[0].set_ylabel('Voltage 1 (V)')
# ax[0].set_ylim(-ylim, ylim)
ax[1].plot(cut_time, cut_v2, linewidth=0.5)
ax[1].axhline(p1_2, c='red', linewidth=0.8)
ax[1].axhline(p99_2, c='red', linewidth=0.8)
ax[1].set_ylabel('Voltage 2 (V)')
ax[1].set_xlabel('Time (s)')
# ax[1].set_ylim(-ylim, ylim)




plt.show()