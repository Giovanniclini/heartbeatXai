
# Heartbeat Classification using CNN-LSTM based on AAMI Guidelines

## Project Overview

This project focuses on classifying heartbeats according to the AAMI guidelines using a CNN-LSTM model. The model utilizes a morphological vector and three additional features: the distance to the previous beat, the distance to the next beat, and the average distance of the last 10 beats. The preprocessing process includes segmentation, denoising, and normalization of ECG signals.

## Datasets

The following datasets were used for training and evaluation:

- **MIT-BIH Arrhythmia Database**: [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **St Petersburg INCART 12-lead Arrhythmia Database**: [PhysioNet St Petersburg INCART Database](https://physionet.org/content/incartdb/1.0.0/)
- **MIT-BIH Supraventricular Arrhythmia Database**: [PhysioNet MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/)

## Preprocessing Pipeline

The preprocessing of ECG signals includes the following steps:

1. **Segmentation**: Identification of heartbeats in the ECG signal.
2. **Denoising**: Noise removal from the signal using the Dual-Tree Complex Wavelet Transform (DTCWT) and baseline wander removal.
3. **Normalization**: Application of MinMax scaling to bring the signal values into a defined range.

### Denoising Details

- **DTCWT Denoising**: Removal of high-frequency noise components while preserving important morphological features of the signal.
- **Baseline Wander Removal**: Elimination of slow variations in the signal, such as those due to respiration, using median filters with window widths specific to P, QRS, and T waves.
- **Normalization**: Scaling the signal to a predetermined range to facilitate model training.

## Model Architecture

The adopted model is a hybrid combination of CNN and LSTM designed to leverage both temporal and spatial features of ECG signals:

- **Convolutional Layers**: Used to extract relevant spatial features from segments of the ECG signal.
- **LSTM Layer**: Processes morphological features to capture temporal dependencies between heartbeats.
- **Integration of RR Features**: The additional RR (Relative R-R intervals) features are concatenated with the LSTM output for further processing.
- **Fully Connected Layers**: Designed to classify heartbeats based on the extracted features.

## Project Structure

```
- checkpoints/
    - best_model.pth
- data/
    - raw_data/
    - processed_data/
        - ecg_training.csv
        - ecg_test.csv
- model/
    - model_handler.py
- dataset/
    - dataset_handler.py
- utils/
    - preprocessing_utils.py
    - denoising_utils.py
- run_preprocess.py
- train.py
- score.py
```


## Results

### Classification Report
              precision    recall  f1-score   support

     Class 0       0.99      0.85      0.92     28065
     Class 1       0.58      0.72      0.64      2297
     Class 2       0.47      0.95      0.62      3271

    accuracy                           0.85     33633
   macro avg       0.68      0.84      0.73     33633
weighted avg       0.91      0.85      0.87     33633

Test Loss: 0.2076
Test Accuracy: 0.8518

### Explanation of Results

- **Class 0** represents normal heartbeats.
- **Class 1** represents supraventricular ectopic beats (SVEB).
- **Class 2** represents ventricular ectopic beats (VEB).

The model demonstrates high performance in identifying normal heartbeats (Class 0), with precision and recall scores of 0.99 and 0.85, respectively. However, there is a significant drop in precision for the arrhythmic classes, particularly Class 1 and Class 2.

In a medical setting, low precision means that a considerable number of non-arrhythmic beats may be misclassified as arrhythmic. This could lead to unnecessary alarms or treatments. Therefore, improving precision, especially for these critical arrhythmia classes, is crucial to avoid misdiagnoses and ensure patient safety.

Despite these challenges, the recall for Class 2 (VEB) is particularly high (0.95), indicating that the model is successful in identifying most arrhythmic beats, though it may generate more false positives.

- **Test Accuracy**: 76.22%

![Confusion Matrix](./images/confusion_matrix_test.png)

## Setup

To install the required dependencies for this project, use the `requirements.txt` file. Follow these steps:

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
1. **Install the dependencies:**:
   ```bash
   pip install -r requirements.txt

## Usage

0. **Data setup:** Make sure to download and save data from the three databases cited at the beginning of the readme in the data directory of the project and to change the paths in the preprocess script accordingly. 
1. **Preprocess the data**: Run `run_preprocess.py` to perform segmentation, denoising, and normalization of the raw data. This script will generate two csv files: ecg_training.csv and ecg_test.csv containing preprocessed data.
2. **Train the model**: Run `train.py` to train the model using the preprocessed data. This script will read the data generated in the previous step.
3. **Test the model**: Use `score.py` to evaluate the model on the test set.

## References

- MIT-BIH Arrhythmia Database: [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- St Petersburg INCART Database: [PhysioNet St Petersburg INCART Database](https://physionet.org/content/incartdb/1.0.0/)
- MIT-BIH Supraventricular Arrhythmia Database: [PhysioNet MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/)
