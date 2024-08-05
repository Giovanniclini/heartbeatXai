def segment_beats(ecg_signal, r_peaks, buff, fs):
    """
    Extracts morphological vectors .

    :param ecg_signal: numpy array of the ECG signal
    :param r_peaks: indices of R peaks in the signal
    :param buff: buffer in seconds
    :param fs: sampling frequency of the ECG signal
    :return: list of numpy arrays, each containing a morphological vector
    """
    vectors = []
    # Time windows in samples
    buffer = int(buff * fs)

    # for 0 to r_peaks - 1 (all index of r_peaks)
    for i in range(0, len(r_peaks)):
        start = r_peaks[i - 1] + buffer if i > 0 else 0
        end = r_peaks[i] + int((r_peaks[i + 1 ] - r_peaks[i])/2) if i < len(r_peaks) - 1 else len(ecg_signal)
        vector = ecg_signal[start:end]
        vectors.append(vector)
    return vectors

