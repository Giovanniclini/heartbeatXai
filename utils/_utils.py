import numpy as np
import wfdb
from wfdb import processing
from ._denoising import normalise_and_denoise_ecg
from sklearn.utils import resample
import csv

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

def extract_data(root, record_list, channel_name, denoise=True, test_record=None):
    """
    Args.
    root: path-to-root-folder
    record_list: list of string containing record name
    channel_name: name of the II channel of ecg
    test_record: selection of record for testing purpose
    """
    filtered_beats_train = []
    filtered_labels_train = []

    filtered_beats_test = []
    filtered_labels_test = []

    pre_rr_train = []
    post_rr_train = []
    avg_rr_past_train = []

    pre_rr_test = []
    post_rr_test = []
    avg_rr_past_test = []

    for rec in record_list:

        if rec in ["102", "104", "107", "217"]:
            continue

        record = wfdb.rdrecord(root + rec)
        annotation = wfdb.rdann(root + rec, 'atr')

        channel_names = record.sig_name
        channel_index = channel_names.index(channel_name)
        ecg_signal = record.p_signal[:, channel_index]

        # Normalizing
        ecg_signal_denoised_normalised = normalise_and_denoise_ecg(ecg_signal, record.fs, denoise=denoise)

        # resample 360hz
        if record.fs != 360:
            record_resampled, annotation_resampled = processing.resample_singlechan(ecg_signal_denoised_normalised, annotation, record.fs, 360)
        else:
            record_resampled = ecg_signal_denoised_normalised
            annotation_resampled = annotation

        beat_locations = annotation_resampled.sample
        beat_symbols = annotation_resampled.symbol
        beats = segment_beats(record_resampled, beat_locations)
        beat_labels = np.array(beat_symbols)

        # Calculate RR features
        pre_rr, post_rr, avg_rr_past = calculate_rr_features(beat_locations)

        
        if rec in test_record:
            filtered_beats_test.extend(beats)
            filtered_labels_test.extend(beat_labels)
            pre_rr_test.extend(pre_rr)
            post_rr_test.extend(post_rr)
            avg_rr_past_test.extend(avg_rr_past)
        else:
            filtered_beats_train.extend(beats)
            filtered_labels_train.extend(beat_labels)
            pre_rr_train.extend(pre_rr)
            post_rr_train.extend(post_rr)
            avg_rr_past_train.extend(avg_rr_past)
            
    filtered_beats_train = np.array(filtered_beats_train, dtype=object)
    filtered_labels_train = np.array(filtered_labels_train)

    filtered_beats_test = np.array(filtered_beats_test, dtype=object)
    filtered_labels_test = np.array(filtered_labels_test)

    pre_rr_train = np.array(pre_rr_train)
    post_rr_train = np.array(post_rr_train)
    avg_rr_past  = np.array(avg_rr_past)

    pre_rr_test = np.array(pre_rr_test)
    post_rr_test = np.array(post_rr_test)
    avg_rr_past_test = np.array(avg_rr_past_test)

    results = {
        'filtered_beats_train': filtered_beats_train,
        'filtered_labels_train': filtered_labels_train,
        'filtered_beats_test': filtered_beats_test,
        'filtered_labels_test': filtered_labels_test,
        'pre_rr_train': pre_rr_train,
        'post_rr_train': post_rr_train,
        'avg_rr_past_train': avg_rr_past_train,
        'pre_rr_test': pre_rr_test,
        'post_rr_test': post_rr_test,
        'avg_rr_past_test': avg_rr_past_test
    }
    
    return results

def calculate_length_thresholds(beats, min_percentile, max_percentile):
    lengths = np.array([len(beat) for beat in beats])
    min_length = np.percentile(lengths, min_percentile)
    max_length = np.percentile(lengths, max_percentile)
    return min_length, max_length

# Function to filter beats by class and length
def filter_by_classes_and_length(beats, labels, pre_rr, post_rr, avg_rr, classes, min_length, max_length):
    valid_lengths_mask = np.array([min_length <= len(beat) <= max_length for beat in beats])
    class_mask = np.isin(labels, classes)
    combined_mask = valid_lengths_mask & class_mask
    return beats[combined_mask], labels[combined_mask], pre_rr[combined_mask], post_rr[combined_mask], avg_rr[combined_mask]

def data_generator(beats, labels, pre_rr, post_rr, avg_rr):
        for beat, label, pre, post, avg in zip(beats, labels, pre_rr, post_rr, avg_rr):
            yield np.hstack((beat, pre, post, avg, label))

def test_data_generator(beats, labels, pre_rr, post_rr, avg_rr):
        for beat, label, pre, post, avg in zip(beats, labels, pre_rr, post_rr, avg_rr):
            yield np.hstack((beat, pre, post, avg, label))

def load_data_from_csv(file_path):
    beats = []
    labels = []
    pre_rr = []
    post_rr = []
    avg_rr = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = np.array(row, dtype=float)
            beats.append(row[:-4])
            pre_rr.append(row[-4])
            post_rr.append(row[-3])
            avg_rr.append(row[-2])
            labels.append(row[-1])

    return np.array(beats), np.array(labels), np.array(pre_rr), np.array(post_rr), np.array(avg_rr)

def downsample_classes(all_padded_beats, all_labels, all_pre_rr, all_post_rr, all_avg_rr, target_sample_count=None):

    unique, counts = np.unique(all_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))

    sorted_class_counts = sorted(class_counts.values(), reverse=True)
    if target_sample_count is None:
        target_sample_count = sorted_class_counts[1]
    
    max_class = max(class_counts, key=class_counts.get)
    
    downsampled_data = {}

    for label in unique:
        X_label = all_padded_beats[all_labels == label]
        pre_rr_label = all_pre_rr[all_labels == label]
        post_rr_label = all_post_rr[all_labels == label]
        avg_rr_label = all_avg_rr[all_labels == label]
        
        if max_class == label:
            X_label_downsampled, y_label_downsampled, pre_rr_downsampled, post_rr_downsampled, avg_rr_downsampled = resample(
                X_label,
                np.full(len(X_label), label),
                pre_rr_label,
                post_rr_label,
                avg_rr_label,
                replace=False,
                n_samples=target_sample_count,
                random_state=42
            )
        else:
            X_label_downsampled, y_label_downsampled, pre_rr_downsampled, post_rr_downsampled, avg_rr_downsampled = (
                X_label, 
                np.full(len(X_label), label), 
                pre_rr_label, 
                post_rr_label, 
                avg_rr_label
            )
        
        downsampled_data[label] = {
            'X': X_label_downsampled,
            'y': y_label_downsampled,
            'pre_rr': pre_rr_downsampled,
            'post_rr': post_rr_downsampled,
            'avg_rr': avg_rr_downsampled
        }
    
    all_padded_resampled_beats = np.concatenate([downsampled_data[label]['X'] for label in unique], axis=0)
    all_padded_resampled_labels = np.concatenate([downsampled_data[label]['y'] for label in unique], axis=0)
    all_resampled_pre_rr = np.concatenate([downsampled_data[label]['pre_rr'] for label in unique], axis=0)
    all_resampled_post_rr = np.concatenate([downsampled_data[label]['post_rr'] for label in unique], axis=0)
    all_resampled_avg_rr = np.concatenate([downsampled_data[label]['avg_rr'] for label in unique], axis=0)
    
    return (all_padded_resampled_beats, 
            all_padded_resampled_labels, 
            all_resampled_pre_rr, 
            all_resampled_post_rr, 
            all_resampled_avg_rr)

def map_to_general_groups(label_vector):
    general_group_mapping = {
        "N": "N",  # Non-ectopic beats (N)
        "L": "N",  
        "R": "N",  
        "j": "N", 
        "e": "N",  
        "A": "S",  # Supraventricular ectopic Beats (S)
        "a": "S",  
        "S": "S",  
        "J": "S",  
        "!": "V",  # Ventricular ectopic Beats (V)
        "V": "V",  
        "E": "V",  
        "[": "V",  
        "]": "V",  
        "F": "F",  # Fusion Beats (F)
        "f": "Q",  # Unknown beats (Q)
        "/": "Q",  
        "Q": "Q"   
    }
    return np.array([general_group_mapping.get(beat, "Q") for beat in label_vector])