import math
import numpy as np
import torch

from scipy.signal.wavelets import cwt, ricker
from scipy.stats import scoreatpercentile


def local_extreme(data, comparator, axis=0, order=1, mode='clip'):
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if ~results.any():
            return results
    return results


def argrelmin(data, axis=0, order=1, mode='clip'):
    """
    Calculate the relative minima of `data`.
    """
    return argrelextrema(data, np.less, axis, order, mode)


def argrelmax(data, axis=0, order=1, mode='clip'):
    """
    Calculate the relative maxima of `data`.
    """
    return argrelextrema(data, np.greater, axis, order, mode)


def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.
    """
    results = local_extreme(data, comparator, axis, order, mode)
    return np.nonzero(results)


def identify_ridge_lines(matr, max_distances, gap_thresh):
    """
    Identify ridges in the 2-D matrix.

    Expect that the width of the wavelet feature increases with increasing row
    number.
    """
    if len(max_distances) < matr.shape[0]:
        raise ValueError('Max_distances must have at least as many rows '
                         'as matr')

    all_max_cols = local_extreme(matr, np.greater, axis=1, order=1)
    # Highest row for which there are any relative maxima
    has_relmax = np.nonzero(all_max_cols.any(axis=1))[0]
    if len(has_relmax) == 0:
        return []
    start_row = has_relmax[-1]
    # Each ridge line is a 3-tuple:
    # rows, cols,Gap number
    ridge_lines = [[[start_row],
                   [col],
                   0] for col in np.nonzero(all_max_cols[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]

        # Increment gap number of each line,
        # set it to zero later if appropriate
        for line in ridge_lines:
            line[2] += 1

        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        # Look through every relative maximum found at current row
        # Attempt to connect them with existing ridge lines.
        for ind, col in enumerate(this_max_cols):
            # If there is a previous ridge line within
            # the max_distance to connect to, do so.
            # Otherwise start a new one.
            line = None
            if len(prev_ridge_cols) > 0:
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
            if line is not None:
                # Found a point close enough, extend current ridge line
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                ridge_lines.append(new_line)

        # Remove the ridge lines with gap_number too high
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines


def filter_ridge_lines(cwt_dat, ridge_lines, window_size=None, min_length=None,
                       min_snr=1, noise_perc=95):
    """
    Filter ridge lines according to prescribed criteria. Intended
    to be used for finding relative maxima.
    """
    num_points = cwt_dat.shape[1]
    if min_length is None:
        min_length = np.ceil(cwt_dat.shape[0] / 4)
    if window_size is None:
        window_size = np.ceil(num_points / 20)

    window_size = int(window_size)
    hf_window, odd = divmod(window_size, 2)

    # Filter based on SNR
    row_one = cwt_dat[0, :]
    noises = np.empty_like(row_one)
    for ind, val in enumerate(row_one):
        window_start = max(ind - hf_window, 0)
        window_end = min(ind + hf_window + odd, num_points)
        noises[ind] = scoreatpercentile(np.abs(row_one[window_start:window_end]),
                                        per=noise_perc)

    def filt_func(line):
        if len(line[0]) < min_length:
            return False
        snr = abs(cwt_dat[line[0][0], line[1][0]] / noises[line[1][0]])
        if snr < min_snr:
            return False
        return True

    return list(filter(filt_func, ridge_lines))


def identify_valley_lines(matr, max_distances, gap_thresh):
    """
    Identify valley in the 2-D matrix.

    Expect that the width of the wavelet feature increases with increasing row
    number.
    """
    if len(max_distances) < matr.shape[0]:
        raise ValueError('Max_distances must have at least as many rows '
                         'as matr')

    all_min_cols = local_extreme(matr, np.less, axis=1, order=1)
    # Highest row for which there are any relative minima
    has_relmin = np.nonzero(all_min_cols.any(axis=1))[0]
    if len(has_relmin) == 0:
        return []
    start_row = has_relmin[-1]
    # Each ridge line is a 3-tuple:
    # rows, cols,Gap number
    valley_lines = [[[start_row],
                    [col],
                    0] for col in np.nonzero(has_relmin[start_row])[0]]
    final_lines = []
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    for row in rows:
        this_min_cols = cols[all_min_cols[row]]

        # Increment gap number of each line,
        # set it to zero later if appropriate
        for line in valley_lines:
            line[2] += 1

        prev_valley_cols = np.array([line[1][-1] for line in valley_lines])
        # Look through every relative maximum found at current row
        # Attempt to connect them with existing ridge lines.
        for ind, col in enumerate(this_min_cols):
            
            line = None
            if len(prev_valley_cols) > 0:
                diffs = np.abs(col - prev_valley_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = valley_lines[closest]
            if line is not None:
                # Found a point close enough, extend current valley line
                line[1].append(col)
                line[0].append(row)
                line[2] = 0
            else:
                new_line = [[row],
                            [col],
                            0]
                valley_lines.append(new_line)

        # Remove the valley lines with gap_number too high
        for ind in range(len(valley_lines) - 1, -1, -1):
            line = valley_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del valley_lines[ind]

    out_lines = []
    for line in (final_lines + valley_lines):
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines


def filter_valley_lines(cwt_dat, valley_lines, min_length=None):
    """
    Filter valley lines according to prescribed criteria. Intended
    to be used for finding relative minima.
    """
    num_points = cwt_dat.shape[1]
    if min_length is None:
        min_length = np.ceil(cwt_dat.shape[0] / 4)

    def filt_func1(line):
        if len(line[0]) < min_length:
            return False
        return True

    return list(filter(filt_func1, valley_lines))


def find_peaks_cwt1(vector, widths, wavelet=None, max_distances=None,
                   gap_thresh=None, min_length=5,
                   min_snr=1, noise_perc=95, window_size=80):
    """
    Find peaks in a 1-D array with wavelet transformation.
    """
    widths = np.array(widths, copy=False, ndmin=1)

    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    if max_distances is None:
        max_distances = widths / 4.0
    if wavelet is None:
        wavelet = ricker

    cwt_dat = cwt(vector, wavelet, widths, window_size=window_size)
    ridge_lines = identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
    filtered = filter_ridge_lines(cwt_dat, ridge_lines, min_length=min_length,
                                  window_size=window_size, min_snr=min_snr,
                                  noise_perc=noise_perc)
    valley_lines = identify_valley_lines(cwt_dat, max_distances, gap_thresh)
    filtered1 = filter_valley_lines(cwt_dat, valley_lines, min_length=min_length)

    max_locs = np.asarray([x[1][0] for x in filtered])
    min_locs = np.asarray([y[1][0] for y in filtered1])
    max_locs.sort()
    min_locs.sort()
    min_end = []
    for i in range(len(max_locs)):
        # c = torch.from_numpy(abs(min_locs - max_locs[i]))
        # index1 = torch.argmin(c)
        # index = index1.numpy()
        index = np.argmin(abs(min_locs - max_locs[i]))
        a = []
        if min_locs[index] - max_locs[i] > 0 and index >= 1:
            a = [min_locs[index - 1], min_locs[index]]
        if min_locs[index] - max_locs[i] < 0 and index < len(min_locs) - 1:
            a = [min_locs[index], min_locs[index + 1]]
        if min_locs[index] - max_locs[i] > 0 and index == 0:
            a = [0, min_locs[index]]
        if min_locs[index] - max_locs[i] < 0 and index == len(min_locs) - 1:
            a = [min_locs[index], len(cwt_dat[0, :]) - 1]
        min_end.append(a)

    return max_locs, min_end


def peaks_position_correction(vector1, vector1_1, peaks, peak_width, intensity_threshold):
    """
    correcting peaks position
    """
    new_peaks = []
    new_peak_width = []
    peak_vector = []
    mz_vector = []
    new_peak_vector = []
    len_peak = len(peaks)
    if len_peak != 0:
        for i in range(len(peaks)):
            p = peaks[i]
            if peak_width[i]:
                w_start = peak_width[i][0]
                w_end = peak_width[i][1]
                w = w_end - w_start
                p_left = vector1[w_start:p]
                p_right = vector1[p + 1:w_end + 1]
                zero_value1 = np.where(p_left == 0)[0] + w_start
                zero_value2 = np.where(p_right == 0)[0] + p + 1
                width1 = w_start
                width2 = w_end
                if len(zero_value1) > 1:
                    zero_value1_diff = np.diff(zero_value1)
                    zero_value1_diff_one = np.where(zero_value1_diff == 1)[0]
                    if len(zero_value1_diff_one) > 0:
                        width_index1 = np.argmax(zero_value1_diff_one)
                        width1 = zero_value1[width_index1 + 1]
                    elif i > 0 and peak_width[i-1]:
                        w_start1 = peak_width[i - 1][0]
                        w_end1 = peak_width[i - 1][1]
                        if w_end1 != w_start:
                            v1 = vector1[w_end1:w_start + 1]
                            zero_value1_1 = np.where(v1 == 0)[0] + w_end1
                            if len(zero_value1_1) > 0 and w_start - zero_value1_1[-1] <= w:
                                width1 = zero_value1_1[-1]
                    elif i == 0:
                        w_end1 = 0
                        if w_end1 != w_start:
                            v1 = vector1[w_end1:w_start + 1]
                            zero_value1_1 = np.where(v1 == 0)[0] + w_end1
                            if len(zero_value1_1) > 0 and w_start - zero_value1_1[-1] <= w:
                                width1 = zero_value1_1[-1]
                if len(zero_value2) > 1:
                    zero_value2_diff = np.diff(zero_value2)
                    zero_value2_diff_one = np.where(zero_value2_diff == 1)[0]
                    if len(zero_value2_diff_one) > 0:
                        width_index2 = np.argmin(zero_value2_diff_one)
                        width2 = zero_value2[width_index2]
                    elif i < len(peaks) - 1 and peak_width[i+1]:
                        w_start2 = peak_width[i + 1][0]
                        w_end2 = peak_width[i + 1][1]
                        if w_start2 != w_end:
                            v2 = vector1[w_end:w_start2 + 1]
                            zero_value2_2 = np.where(v2 == 0)[0] + w_end
                            if len(zero_value2_2) > 0 and zero_value2_2[0] - w_end <= w:
                                width2 = zero_value2_2[0]
                    elif i == len(peaks) - 1:
                        w_start2 = len(vector1)
                        if w_start2 != w_end:
                            v2 = vector1[w_end:w_start2]
                            zero_value2_2 = np.where(v2 == 0)[0] + w_end
                            if len(zero_value2_2) > 0 and zero_value2_2[0] - w_end <= w:
                                width2 = zero_value2_2[0]
                if len(zero_value1) <= 1 and i > 0 and peak_width[i-1]:
                    w_start1 = peak_width[i - 1][0]
                    w_end1 = peak_width[i - 1][1]
                    if w_end1 != w_start:
                        v1 = vector1[w_end1:w_start + 1]
                        zero_value1_1 = np.where(v1 == 0)[0] + w_end1
                        if len(zero_value1_1) > 0 and w_start - zero_value1_1[-1] <= w:
                            width1 = zero_value1_1[-1]
                if len(zero_value1) <= 1 and i == 0:
                    w_end1 = 0
                    if w_end1 != w_start:
                        v1 = vector1[w_end1:w_start + 1]
                        zero_value1_1 = np.where(v1 == 0)[0] + w_end1
                        if len(zero_value1_1) > 0 and w_start - zero_value1_1[-1] <= w:
                            width1 = zero_value1_1[-1]
                if len(zero_value2) <= 1 and i < len(peaks) - 1 and peak_width[i+1]:
                    w_start2 = peak_width[i + 1][0]
                    w_end2 = peak_width[i + 1][1]
                    if w_start2 != w_end:
                        v2 = vector1[w_end:w_start2 + 1]
                        zero_value2_2 = np.where(v2 == 0)[0] + w_end
                        if len(zero_value2_2) > 0 and zero_value2_2[0] - w_end <= w:
                            width2 = zero_value2_2[0]
                if len(zero_value2) <= 1 and i == len(peaks) - 1:
                    w_start2 = len(vector1)
                    if w_start2 != w_end:
                        v2 = vector1[w_end:w_start2]
                        zero_value2_2 = np.where(v2 == 0)[0] + w_end
                        if len(zero_value2_2) > 0 and zero_value2_2[0] - w_end <= w:
                            width2 = zero_value2_2[0]
                v = vector1[width1:width2 + 1]
                exist = (v > 0) * 1.0
                factor = np.ones(len(v))
                res = np.dot(exist, factor)
                if res >= 5 and max(v) > intensity_threshold:
                    index1 = np.argmax(v) + width1
                    new_peaks.append(index1)
                    aa = [width1, width2]
                    new_peak_width.append(aa)
                    peak_vector.append(vector1[width1:width2 + 1])
                    mz_vector.append(vector1_1[width1:width2 + 1])

    return new_peaks, new_peak_width, peak_vector, mz_vector


def extend_peaks0(vector2, peaks2, peak_width2):
    new_peaks1 = []
    new_peak_width1 = []
    peak_vector1 = []
    len_col = len(vector2)
    len_peak2 = len(peaks2)
    if len_peak2 != 0:
        for i in range(len(peaks2)):
            window = peak_width2[i][1] - peak_width2[i][0]
            w_left = int(max([peak_width2[i][0] - window, 0]))
            w_right = int(min([peak_width2[i][1] + window, len_col]))
            new_peak_width1.append([w_left, w_right])
            peak_vector1.append(vector2[w_left:w_right + 1])

    return new_peak_width1, peak_vector1

