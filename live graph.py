import sys, serial
import numpy as np
import os
from time import sleep
import matplotlib
from collections import deque
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from pandas import DataFrame as df
#import ble


# class that holds analog data for N samples

class AnalogData:

    # constr
    def __init__(self, maxLen):
        self.ax = deque([0.0] * maxLen)
        self.ay = deque([0.0] * maxLen)
        self.maxLen = maxLen

    # ring buffer
    def addToBuf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert (len(data) == 2)
        self.addToBuf(self.ax, data[0])
        self.addToBuf(self.ay, data[1])


# plot class
class AnalogPlot:
    # constr
    def __init__(self, analogData):
        # set plot to animated
        plt.ion()
        self.axline, = plt.plot(analogData.ax)
        self.ayline, = plt.plot(analogData.ay)
        plt.ylim([-180, 180])

    # update plot
    def update(self, analogData):
        self.axline.set_ydata(analogData.ax)
        self.ayline.set_ydata(analogData.ay)
        plt.draw()
        plt.pause(0.0001)


# checkfile(path) function to write file
def checkfile(path):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        return path

    root, ext = os.path.splitext(os.path.expanduser(path))
    dir = os.path.dirname(root)
    fname = os.path.basename(root)
    candidate = fname + ext
    index = 0
    ls = set(os.listdir(dir))
    while candidate in ls:
        candidate = "{}_{}{}".format(fname, index, ext)
        index += 1
    return os.path.join(dir, candidate)


# prepare data for export function
def prepareData(arduinoString, dataLabel):
    dataArray = arduinoString.decode().split(',')
    dataValue = np.zeros((1, 13))
    dataValue[0, 0] = float(dataArray[1])
    for i in range(3):
        dataValue[0, i + 1] = float(dataArray[i + 3])
        dataValue[0, i + 4] = float(dataArray[i + 7])
        dataValue[0, i + 7] = float(dataArray[i + 11])
        dataValue[0, i + 10] = float(dataArray[i + 15])
    toDf = df(dataValue, columns=dataLabel)
    return (toDf)


# main() function
def main():
    # expects 1 arg - serial port string
    #  if(len(sys.argv) != 2):
    #    print ('Example usage: python showdata.py "/dev/tty.usbmodem411"')
    #    exit(1)

    strPort = '/dev/cu.usbmodem1411'
    # strPort = sys.argv[1];
    path = '/Users/Sam/Dropbox/Python/EMG Data/temp/out.csv'
    path = checkfile(path)
    # plot parameters
    analogData = AnalogData(100)
    analogPlot = AnalogPlot(analogData)
    # setup label for data
    dataLabel = ['angle',
                 'acce1_X', 'acce1_Y', 'acce1_Z',
                 'gyro1_X', 'gyro1_Y', 'gyro1_Z',
                 'acce2_X', 'acce2_Y', 'acce2_Z',
                 'gyro2_X', 'gyro2_Y', 'gyro2_Z']
    final_df = df(columns=dataLabel)

    print('plotting data...')

    # open serial port
    ser = serial.Serial(strPort, 9600)
    while True:
        try:
            line = ser.readline()
            toDf = prepareData(line, dataLabel)
            final_df = final_df.append(toDf)
            data = [toDf["angle"].iloc[-1], 0]
            # print data
            analogData.add(data)
            analogPlot.update(analogData)
        except (KeyboardInterrupt, SystemExit):
            print('exiting, exporting data')
            final_df.to_csv(path, index=False, header=False)
            print('finished')
            break
    # # close serial

    ser.flush()
    ser.close()


# call main
if __name__ == '__main__':
    main()
