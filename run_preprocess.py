from utils._utils import extract_data, calculate_length_thresholds, filter_by_classes_and_length, data_generator, test_data_generator, map_to_general_groups, downsample_classes
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
import wfdb
import csv
import os

if __name__ == '__main__':

    record_list_mitbih = wfdb.get_record_list('mitdb')
    record_list_stpter = wfdb.get_record_list('incartdb')
    record_list_supra = wfdb.get_record_list('svdb')

    #MITBIH Arrythmia database
    root = 'data/mit-bih-arrhythmia-database-1.0.0/'
    test_records = ["234", "233", "232", "231", "230"]
    results_mitbih = extract_data(root, record_list_mitbih, 'MLII', True, test_records)
    results_mitbih_noise = extract_data(root, record_list_mitbih, 'MLII', False, test_records)

    #INCART database
    root = 'data/files/'
    test_records = ["I75", "I74", "I73", "I72", "I71", "I70"]
    results_stpter = extract_data(root, record_list_stpter, 'II', True, test_records)
    results_stpter_noise = extract_data(root, record_list_stpter, 'II', False, test_records)

    #MITBIH Supraventricular Arrythmia database
    root = 'data/mit-bih-supraventricular-arrhythmia-database-1.0.0/'
    test_records = ["894", "893", "892", "891", "890"]
    results_supra = extract_data(root, record_list_supra, 'ECG2', True, test_records)
    results_supra_noise = extract_data(root, record_list_supra, 'ECG2', False, test_records)

    all_beats = np.concatenate((results_mitbih['filtered_beats_train'], results_stpter['filtered_beats_train'], results_supra['filtered_beats_train'],
                                results_mitbih_noise['filtered_beats_train'], results_stpter_noise['filtered_beats_train'], results_supra_noise['filtered_beats_train']), axis=0)
    beats_test = np.concatenate((results_mitbih_noise['filtered_beats_test'], results_stpter_noise['filtered_beats_test'], results_supra_noise['filtered_beats_test']), axis=0)

    all_pre_rr = np.concatenate((results_mitbih['pre_rr_train'], results_stpter['pre_rr_train'], results_supra['pre_rr_train'],
                                results_mitbih_noise['pre_rr_train'], results_stpter_noise['pre_rr_train'], results_supra_noise['pre_rr_train']), axis=0)
    all_post_rr = np.concatenate((results_mitbih['post_rr_train'], results_stpter['post_rr_train'], results_supra['post_rr_train'],
                                    results_mitbih_noise['post_rr_train'], results_stpter_noise['post_rr_train'], results_supra_noise['post_rr_train']), axis=0)
    all_avg_rr = np.concatenate((results_mitbih['avg_rr_past_train'], results_stpter['avg_rr_past_train'], results_supra['avg_rr_past_train'],
                                    results_mitbih_noise['avg_rr_past_train'], results_stpter_noise['avg_rr_past_train'], results_supra_noise['avg_rr_past_train']), axis=0)

    all_labels = np.concatenate((results_mitbih['filtered_labels_train'], results_stpter['filtered_labels_train'], results_supra['filtered_labels_train'],
                                    results_mitbih_noise['filtered_labels_train'], results_stpter_noise['filtered_labels_train'], results_supra_noise['filtered_labels_train']), axis=0)
    labels_test = np.concatenate((results_mitbih_noise['filtered_labels_test'], results_stpter_noise['filtered_labels_test'], results_supra_noise['filtered_labels_test']), axis=0)

    all_pre_rr_test = np.concatenate((results_mitbih_noise['pre_rr_test'], results_stpter_noise['pre_rr_test'], results_supra_noise['pre_rr_test']), axis=0)
    all_post_rr_test = np.concatenate((results_mitbih_noise['post_rr_test'], results_stpter_noise['post_rr_test'], results_supra_noise['post_rr_test']), axis=0)
    all_avg_rr_test = np.concatenate((results_mitbih_noise['avg_rr_past_test'], results_stpter_noise['avg_rr_past_test'], results_supra_noise['avg_rr_past_test']), axis=0)

    all_labels = map_to_general_groups(all_labels)
    labels_test = map_to_general_groups(labels_test)

    # Desired classes
    desired_classes = ['N', 'S', 'V']

    # Define the percentage thresholds
    min_percentile = 5  # 5th percentile
    max_percentile = 95  # 95th percentile

    # Calculate length thresholds based on the training data
    min_length, max_length = calculate_length_thresholds(all_beats, min_percentile, max_percentile)

    # Apply filtering on training data
    all_beats, all_labels, all_pre_rr, all_post_rr, all_avg_rr = filter_by_classes_and_length(
        all_beats, all_labels, all_pre_rr, all_post_rr, all_avg_rr, desired_classes, min_length, max_length
    )

    # Apply filtering on testing data using the same length thresholds
    beats_test, labels_test, all_pre_rr_test, all_post_rr_test, all_avg_rr_test = filter_by_classes_and_length(
        beats_test, labels_test, all_pre_rr_test, all_post_rr_test, all_avg_rr_test, desired_classes, min_length, max_length
    )

    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)
    labels_test = label_encoder.transform(labels_test)

    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Label Mapping:", label_mapping)

    all_padded_beats = pad_sequences(all_beats, padding='post', dtype='float32')
    padded_beats_test = pad_sequences(beats_test, padding='post', dtype='float32')

    resampled_data = downsample_classes(all_padded_beats, all_labels, all_pre_rr, all_post_rr, all_avg_rr)
    all_padded_resampled_beats, all_padded_resampled_labels, all_resampled_pre_rr, all_resampled_post_rr, all_resampled_avg_rr = resampled_data

    with open(os.path.join('data', 'ecg_training.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_generator(all_padded_resampled_beats, all_padded_resampled_labels.reshape(-1, 1),
                                all_resampled_pre_rr, all_resampled_post_rr, all_resampled_avg_rr):
            writer.writerow(row)

    with open(os.path.join('data', 'ecg_test.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in test_data_generator(padded_beats_test, labels_test.reshape(-1, 1),
                                    all_pre_rr_test, all_post_rr_test, all_avg_rr_test):
            writer.writerow(row)
