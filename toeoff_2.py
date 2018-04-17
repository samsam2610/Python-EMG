import numpy as np  # Import numpy
from scipy.fftpack import fft, fftshift
from scipy import signal
from scipy.signal import butter, lfilter, hilbert
import matplotlib.pyplot as plt
import pandas as pd
from drawnow import *
from collections import deque

input_data = '/Users/Sam/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Folder/Final HSS Data 1_16_18/Main_Test_1_fast_walk_4__Rep_1.50.csv'

data = pd.read_csv(input_data)
string_part = ["TIBIALIS", "Gyro X"]
columns_label = [col for col in data.columns if all(c in col for c in string_part)]
columns_label.insert(0, "X[s].33")
imu_Data = data.loc[:, columns_label]
imu_Data = imu_Data.loc[(imu_Data["X[s].33"] < 8)]

diff = imu_Data.diff()
index_record = []
time_record = 0
pre_value = 0
cur_value = 0
cur_diff = 0

for value in imu_Data.itertuples():
    elapsed_time = value[1] - time_record
    cur_value = value[2]
    cur_diff = cur_value - pre_value
    if imu_Data.iloc[value[0], 1] < -200 \
            and elapsed_time > 0.5 \
            and cur_diff * pre_diff < 0:
        index_record.append(value[0])
        time_record = value[1]
        print("added")

    pre_value = cur_value
    pre_diff = cur_diff

fig, ax = plt.subplots(1, 1, figsize=[15, 15])
x = imu_Data.iloc[0:, 0]
y = imu_Data.iloc[0:, 1]
ax.plot(x, y)

for index in index_record:
    value = imu_Data.iloc[index, 0]
    ax.axvline(x=value, color='#d62728')

plt.title('Gyroscope X')
plt.xlabel('Time (s)')
plt.ylabel('deg/s')
plt.show()