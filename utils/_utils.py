import numpy as np

def segment_beats(ecg_signal, r_peaks):
    """
    Segment beats.

    :param ecg_signal: numpy array of the ECG signal
    :param r_peaks: indices of R peaks in the signal
    :return: list of numpy arrays, each containing a morphological vector
    """
    vectors = []

    for i in range(0, len(r_peaks)):
        start = r_peaks[i - 1] + int((r_peaks[i] - r_peaks[i - 1])/4) if i > 0 else 0
        end = r_peaks[i] + int((r_peaks[i + 1 ] - r_peaks[i])/2) if i < len(r_peaks) - 1 else len(ecg_signal)
        vector = ecg_signal[start:end]
        vectors.append(vector)
    return vectors

def calculate_rr_features(beat_locations):
    """
    Extract additional features.

    For each beat extract: 
    1. Distance with the previous beat;
    2. Distance with the following beat;
    3. Avg distance of the last 10 beats.
    """
    pre_rr = []
    post_rr = []
    avg_rr_past = []

    beat_locations = np.array(beat_locations)

    for i in range(len(beat_locations)):
        if i == 0:
            pre_rr.append(beat_locations[1] - beat_locations[0])
        else:
            pre_rr.append(beat_locations[i] - beat_locations[i-1])

        if i == len(beat_locations) - 1:
            post_rr.append(beat_locations[-1] - beat_locations[-2])
        else:
            post_rr.append(beat_locations[i+1] - beat_locations[i])

        if i == 0:
            avg_rr_past.append(pre_rr[-1])
        else:
            avg_rr_past.append(np.mean(np.diff(beat_locations[max(0, i-9):i+1])))

    pre_rr = np.array(pre_rr)
    post_rr = np.array(post_rr)
    avg_rr_past = np.array(avg_rr_past)

    pre_rr = np.nan_to_num(pre_rr, nan=np.nanmean(pre_rr))
    post_rr = np.nan_to_num(post_rr, nan=np.nanmean(post_rr))
    avg_rr_past = np.nan_to_num(avg_rr_past, nan=np.nanmean(avg_rr_past))

    return pre_rr, post_rr, avg_rr_past