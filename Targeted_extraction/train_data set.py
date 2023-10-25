import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import savemat

from parse_data import get_EICs1, peak_detection, cnn_data_preprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get user-specified data
    parent = os.path.realpath('example.mzML')
    file = os.path.basename(parent)
    csv = pd.read_csv('example_ mz.csv')
    mz_list = csv['m/z'].tolist()
    eics, mz_value_1 = get_EICs1(file, mz_list, delta_mz=0.005)
    peaks, peak_widths_end, mz_end, cnn_data = peak_detection(eics, mz_value_1, intensity_threshold=100)
    cnn_data_end = cnn_data_preprocess(cnn_data)
    data_end1 = []
    for i in range(len(cnn_data_end)):
        data1 = cnn_data_end[i]
        for j in range(len(data1)):
            data2 = data1[j]
            data_end1.append(data2)
    savemat('train_data.mat', {'signal': data_end1})
    xx = len(data_end1[0])
    x = []
    aa = 0
    for i in range(0, xx):
        x.append(i)
    for j in range(len(data_end1)):
        y = data_end1[j]
        plt.plot(x, y)
        plt.xlabel('scans number')
        plt.ylabel('Intensity')
        # Path for saving EIC images
        plt.savefig("photo/model_{:06}.jpg".format(aa))
        plt.clf()
        aa += 1


if __name__ == '__main__':
    main()
