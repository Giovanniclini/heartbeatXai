from utils._utils import extract_data, calculate_length_thresholds, filter_by_classes_and_length, data_generator, test_data_generator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
import numpy as np
import wfdb
import csv

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

    # Desired classes
    desired_classes = ['N', 'S', 'V', 'F']

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

    X_label_0 = all_padded_beats[all_labels == 0]
    X_label_1 = all_padded_beats[all_labels == 1]
    X_label_2 = all_padded_beats[all_labels == 2]
    X_label_3 = all_padded_beats[all_labels == 3]
    X_label_4 = all_padded_beats[all_labels == 4]

    pre_rr_0 = all_pre_rr[all_labels == 0]
    pre_rr_1 = all_pre_rr[all_labels == 1]
    pre_rr_2 = all_pre_rr[all_labels == 2]
    pre_rr_3 = all_pre_rr[all_labels == 3]
    pre_rr_4 = all_pre_rr[all_labels == 4]

    post_rr_0 = all_post_rr[all_labels == 0]
    post_rr_1 = all_post_rr[all_labels == 1]
    post_rr_2 = all_post_rr[all_labels == 2]
    post_rr_3 = all_post_rr[all_labels == 3]
    post_rr_4 = all_post_rr[all_labels == 4]

    avg_rr_0 = all_avg_rr[all_labels == 0]
    avg_rr_1 = all_avg_rr[all_labels == 1]
    avg_rr_2 = all_avg_rr[all_labels == 2]
    avg_rr_3 = all_avg_rr[all_labels == 3]
    avg_rr_4 = all_avg_rr[all_labels == 4]

    X_label_1_downsampled, y_label_1_downsampled, pre_rr_1_downsampled, post_rr_1_downsampled, avg_rr_1_downsampled = resample(
        X_label_1,
        np.full(len(X_label_1), 1),
        pre_rr_1,
        post_rr_1,
        avg_rr_1,
        replace=False,
        n_samples=60000,
        random_state=42
    )

    all_padded_resampled_beats = np.concatenate((X_label_0, X_label_1_downsampled, X_label_2, X_label_3, X_label_4), axis=0)
    all_padded_resampled_labels = np.concatenate((
        np.full(len(X_label_0), 0),
        y_label_1_downsampled,
        np.full(len(X_label_2), 2),
        np.full(len(X_label_3), 3),
        np.full(len(X_label_4), 4)
    ), axis=0)

    all_resampled_pre_rr = np.concatenate((pre_rr_0, pre_rr_1_downsampled, pre_rr_2, pre_rr_3, pre_rr_4), axis=0)
    all_resampled_post_rr = np.concatenate((post_rr_0, post_rr_1_downsampled, post_rr_2, post_rr_3, post_rr_4), axis=0)
    all_resampled_avg_rr = np.concatenate((avg_rr_0, avg_rr_1_downsampled, avg_rr_2, avg_rr_3, avg_rr_4), axis=0)

    with open('data\ecg_training.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_generator(all_padded_resampled_beats, all_padded_resampled_labels.reshape(-1, 1),
                                all_resampled_pre_rr, all_resampled_post_rr, all_resampled_avg_rr):
            writer.writerow(row)

    with open('data\ecg_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in test_data_generator(padded_beats_test, labels_test.reshape(-1, 1),
                                    all_pre_rr_test, all_post_rr_test, all_avg_rr_test):
            writer.writerow(row)
