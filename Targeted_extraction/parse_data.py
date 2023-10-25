import os
import json
import pymzml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.interpolate import interp1d
from peak_detection1 import find_peaks_cwt1, peaks_position_correction, extend_peaks0


def get_EICs(path, target_mz, mode, delta_mz=0.01):
    """
    :param path: path to mzml file
    :param target_mz: m/z
    :param delta_mz: m/z tolerance
    :param mode: positive:1, negative:0
    :return: EICs - a list of EIC objects found in current file
    """
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    scans = []
    scan_time_1 = []
    for scan in run:
        if scan.ms_level == 1 and scan.polarity == mode:
            scans.append(scan)
            scan_time_1.append(scan.scan_time[0])
    length_scans = len(scans)
    points = []
    mz_value = []

    init_i = np.zeros(length_scans)
    init_mz = np.zeros(length_scans)
    for k in range(length_scans):
        init_scan = scans[k]
        intensity = []
        mz = []
        scan_time_value = []
        if min(init_scan.mz) - delta_mz <= target_mz <= max(init_scan.mz) + delta_mz:
            cha = init_scan.mz - target_mz
            cha_new = np.where(abs(cha) <= delta_mz)[0]
            if len(cha_new) > 0:
                for a in range(len(cha_new)):
                    intensity.append(init_scan.i[cha_new[a]])
                    mz.append(init_scan.mz[cha_new[a]])
                init_i[k] = sum(intensity)
                init_mz[k] = sum(mz) / len(cha_new)
    EICs = init_i
    mz_value = init_mz
    return EICs, mz_value, scan_time_1


def get_EICs1(path, target_mz, mode, delta_mz=0.01):
    """
    :param path: path to mzml file
    :param target_mz: m/z list
    :param delta_mz: m/z tolerance
    :param mode: positive:1, negative:0
    :return: EICs - a list of EIC objects found in current file
    """
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    scans = []
    scan_time_1 = []
    scan_time_2 = []
    scans_2 = []
    scan_precursor_mz_2 = []
    scan_precursor_i_2 = []
    scan_mz_2 = []
    scan_i_2 = []
    for scan in run:
        if scan.ms_level == 1 and scan.polarity == mode:
            scans.append(scan)
            scan_time_1.append(scan.scan_time[0])
        if scan.ms_level == 2 and scan.polarity == mode:
            scan_time_2.append(scan.scan_time[0])
            scan_precursor_mz_2.append(scan.selected_precursors[0]['mz'])
            scan_precursor_i_2.append(scan.selected_precursors[0]['i'])
            scan_mz_2.append(scan.mz)
            scan_i_2.append(scan.i)
    length_scans = len(scans)
    EICs = []  # completed EICs
    mz_value = []

    for k in range(len(target_mz)):
        init_i = np.zeros(length_scans)
        init_mz = np.zeros(length_scans)
        for kk in range(length_scans):
            init_scan = scans[kk]
            intensity = []
            mz = []
            scan_time_value = []
            if min(init_scan.mz) - delta_mz <= target_mz[k] <= max(init_scan.mz) + delta_mz:
                cha = init_scan.mz - target_mz[k]
                cha_new = np.where(abs(cha) <= delta_mz)[0]
                if len(cha_new) > 0:
                    for a in range(len(cha_new)):
                        intensity.append(init_scan.i[cha_new[a]])
                        mz.append(init_scan.mz[cha_new[a]])
                    init_i[kk] = sum(intensity)
                    init_mz[kk] = sum(mz) / len(cha_new)
        EICs.append(init_i)
        mz_value.append(init_mz)
    return EICs, mz_value, scan_time_1, scan_time_2, scan_precursor_mz_2, scan_precursor_i_2, scan_mz_2, scan_i_2


def peak_detection1(list1, list1_1, intensity_threshold):
    peaks1 = []
    peaks = []
    peak_widths_end1 = []
    peak_widths_end = []
    peak_widths_end3 = []
    mz_end = []
    cnn_data = []
    for i in range(len(list1)):
        peak_inds1, peak_endpoints1 = find_peaks_cwt1(list1[i], np.arange(1, 40))

        peak_inds2, peak_endpoints2, peak_vector2, mz_vector2 = peaks_position_correction(list1[i], list1_1[i],
                                                                                          peak_inds1, peak_endpoints1,
                                                                                          intensity_threshold)
        peak_endpoints3, peak_vector3 = extend_peaks0(list1[i], peak_inds2, peak_endpoints2)

        peaks1.append(peak_inds1)
        peaks.append(peak_inds2)
        peak_widths_end1.append(peak_endpoints1)
        peak_widths_end.append(peak_endpoints2)
        peak_widths_end3.append(peak_endpoints3)
        mz_end.append(mz_vector2)
        cnn_data.append(peak_vector3)
    return peaks1, peaks, peak_widths_end1, peak_widths_end, peak_widths_end3, mz_end, cnn_data


def normalization(vector, points):
    data1 = max(vector)
    if data1 != 0:
        vector = vector / data1
        interpolate = interp1d(np.arange(len(vector)), vector, kind='linear')
        new_vector = interpolate(np.arange(points) / (points - 1) * (len(vector) - 1))
    else:
        data1 = 1
        vector = vector / data1
        interpolate = interp1d(np.arange(len(vector)), vector, kind='linear')
        new_vector = interpolate(np.arange(points) / (points - 1) * (len(vector) - 1))

    return new_vector


def cnn_data_preprocess1(peaks1, peaks, peak_widths_end1, peak_widths_end, peak_widths_end3, cnn_data):
    peaks1_end2 = []
    peaks_end2 = []
    widths1_end2 = []
    widths_end2 = []
    widths3_end2 = []
    data_end2 = []
    for i in range(len(cnn_data)):
        peaks1_end1 = []
        peaks_end1 = []
        widths1_end1 = []
        widths_end1 = []
        widths3_end1 = []
        data_end1 = []
        data1 = peaks1[i]
        data2 = peaks[i]
        data3 = peak_widths_end1[i]
        data4 = peak_widths_end[i]
        data5 = peak_widths_end3[i]
        data6 = cnn_data[i]
        for j in range(len(data6)):
            data1_1 = data1[j]
            data2_2 = data2[j]
            data3_3 = data3[j]
            data4_4 = data4[j]
            data5_5 = data5[j]
            data6_6 = data6[j]
            peaks1_end1.append(data1_1)
            peaks_end1.append(data2_2)
            widths1_end1.append(data3_3)
            widths_end1.append(data4_4)
            widths3_end1.append(data5_5)
            data_end1.append(data6_6)
        peaks1_end2.append(peaks1_end1)
        peaks_end2.append(peaks_end1)
        widths1_end2.append(widths1_end1)
        widths_end2.append(widths_end1)
        widths3_end2.append(widths3_end1)
        data_end2.append(data_end1)

    return peaks1_end2, peaks_end2, widths1_end2, widths_end2, widths3_end2, data_end2


def cnn_data_preprocess0(cnn_data):
    data_end2 = []
    for i in range(len(cnn_data)):
        data_end1 = []
        data1 = cnn_data[i]
        for j in range(len(data1)):
            data2 = data1[j]
            data = normalization(data2, 128)
            data_end1.append(data)
        data_end2.append(data_end1)

    return data_end2


def cnn_data_preprocess(cnn_data):
    data = []
    if len(cnn_data) > 0:
        data = normalization(cnn_data, 128)
    return data
