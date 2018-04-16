 


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

del data

order = 5
fs = 2000
cutoff = 10
second_value = 0
previous_caching_value = 0
receiving_caching_value = 0
previous_value = 0
count = 0
loop_count = 0
index_record = []

def peak_counting(min_bound, max_bound, new_value, old_value, step, sign):

    if new_value * old_value < 0 and \
            (new_value - old_value) * sign < 0 and \
            min_bound < abs(step) < max_bound:
        return True
    else:
        return False


diff = imu_Data.diff()

for value in imu_Data.itertuples():

    receiving_value = value._2
    diff_value = receiving_value - previous_value
    previous_value = receiving_value
    receiving_caching_value = diff_value
    min_diff_value = 5
    max_diff_value = 360
    check = peak_counting(min_diff_value,
                          max_diff_value,
                          receiving_caching_value,
                          previous_caching_value,
                          diff_value,
                          -1)
    if check is True:
        index_record.append(value._1)
    previous_caching_value = receiving_caching_value

size = 15
min_distance = 75
max_distance = 100
data_pack = deque([0]*size)
data_pack_index = deque([0]*size)
count_toe_off = 0
count_heel_strike = 0
index_toe_off_record = []
index_heel_strike_record = []


def check_peak_diff(data: deque, min_distance, max_distance, count_1, count_2):
    index_max = data.index(max(data))
    index_min = data.index(min(data))
    count_1_index = 0
    count_2_index = 0

    if min_distance <= max(data) - min(data) <= max_distance:
        if index_max > index_min:
            count_1_index = index_max
            for index in range(index_min, index_max+1):
                data[index] = np.nan
            count_1 += 1
        else:
            count_2_index = index_min
            for index in range(index_max, index_min+1):
                data[index] = np.nan
            count_2 += 1

    return data, count_1, count_2, count_1_index, count_2_index


for value in diff.itertuples():

    data_pack.pop()
    data_pack_index.pop()
    data_pack.insert(0, value._2)
    data_pack_index.insert(0, value.Index)

    print(data_pack)
    data_pack, \
        count_toe_off, \
        count_heel_strike,\
        index_toe_off,\
        index_heel_strike = check_peak_diff(data_pack, min_distance, max_distance, count_toe_off, count_heel_strike)

    if index_toe_off is not 0:
        index_toe_off_record.append(data_pack_index[index_toe_off])

    if index_heel_strike is not 0:
        index_heel_strike_record.append(data_pack_index[index_heel_strike])


fig, ax = plt.subplots(2, 1, figsize=[15, 15])
x = imu_Data.iloc[0:, 0]
y = imu_Data.iloc[0:, 1]
ax[0].plot(x, y)

x = imu_Data.iloc[1:, 0]
y = diff.iloc[1:, 1]
ax[1].plot(x, y)


for index in index_toe_off_record:
    value = imu_Data.iloc[index+5, 0]
    ax[0].axvline(x=value, color='#d62728')


for index in index_heel_strike_record:
    value = imu_Data.iloc[index+2, 0]
    ax[0].axvline(x=value, color='#45F112')

plt.show()
