
import numpy as np  # Import numpy
from scipy.fftpack import fft, fftshift
from scipy import signal
from scipy.signal import butter, lfilter, hilbert
import matplotlib.pyplot as plt
import pandas as pd
from drawnow import *


order= 5
fs = 2000
cutoff = 10

input_data = '/Users/Sam/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Final HSS Data 1_16_18/Main_Test_1_fast_walk_4__Rep_1.50.csv'

data = pd.read_csv(input_data)
columns_label = [col for col in data.columns if "TIBIALIS" in col]
columns_label.insert(0, "X[s].33")
imu_Data = data.loc[:, columns_label]
imu_Data = imu_Data.loc[(imu_Data["X[s].33"] < 8)]

del data

fig, ax = plt.subplots(imu_Data.shape[1]-1, figsize=[15, 15])
for i in range(imu_Data.shape[1]-1):
    x = imu_Data.iloc[1:, 0]
    y = imu_Data.iloc[1:, i+1]
    plot_name = imu_Data.columns[i+1]
    ax[i].plot(x, y)
    ax[i].set_title(plot_name)
    del x, y

plt.tight_layout()
plt.show()
