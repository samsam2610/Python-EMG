#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:43:32 2018

@author: Sam
"""


import numpy as np  # Import numpy
from scipy.fftpack import fft, fftshift
from scipy import signal
from scipy.signal import butter, lfilter, hilbert
import matplotlib.pyplot as plt  # import matplotlib library
import pandas as pd
from drawnow import *

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def frequency_domain(data,nfft):
    freq_data = fft(data, nfft)
    freq_data = fftshift(freq_data)
    return freq_data

order= 5
fs = 2000
cutoff = 10

input_data = '/Users/Sam/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Final HSS Data 1_16_18/Main_Test_1_fast_walk_4__Rep_1.50.csv'

data = pd.read_csv(input_data)
emg_Data = data.iloc[0:, 1]
emg_1s_Data = data.iloc[0:, 0:2]
del data

emg_Data.columns = ['EMG']

# Convert Data Frame to numpy array
emg_Data = emg_Data.as_matrix()
emg_1s_Data = emg_1s_Data.as_matrix()

# Indexing array

emg_1s_Data = emg_1s_Data[(emg_1s_Data[0:, 0] < 2) & (emg_1s_Data[0:, 0] > 1)]
emg_Data = emg_1s_Data[0:, 1]

emg_Data_envelope_raw = abs(hilbert(emg_Data))

emg_Data_highpass = signal.detrend(abs(emg_Data))
emg_Data_filtered = butter_lowpass_filter(emg_Data_highpass, cutoff, fs, order)
emg_Data_envelope = np.abs(hilbert(emg_Data_filtered))

nfft = emg_Data.size

emg_Freq = frequency_domain(emg_Data, nfft)
emg_Freq_filtered = frequency_domain(emg_Data_filtered, nfft)

nVals = np.arange(0, nfft)
nVals = nVals/nfft

fVals = np.arange(-nfft/2, nfft/2)
fVals = fs*fVals/nfft


fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(9, 9))
ax = axs[0, 0]
ax.plot(nVals, emg_Data_filtered)
ax.set_title("Filtered EMG Data")

ax = axs[0, 1]
ax.plot(nVals, emg_Data)
ax.set_title("Raw EMG Data")

ax = axs[1, 0]
ax.plot(fVals, abs(emg_Freq_filtered))
ax.set_title("Filtered EMG Data Freq Domain")
ax.set_xlim(-10, 10)

ax = axs[1, 1]
ax.plot(fVals, abs(emg_Freq))
ax.set_title("Raw EMG Data Freq Domain")

ax = axs[2, 0]
ax.plot(fVals, emg_Data_envelope)
ax.set_title("Envelope Filtered EMG Data")

ax = axs[2,1]
ax.plot(fVals, emg_Data_envelope_raw)
ax.set_title("Envelope Raw EMG Data")

fig.tight_layout()
fig.show()

#if __name__ == '__main__':
#    main()
