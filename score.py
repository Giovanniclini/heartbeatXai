from utils._utils import load_data_from_csv
from dataset.ECGDatasetHandler import ECGDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from model.model import CNN_LSTM_Model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

if __name__ == '__main__':

    padded_beats_test, labels_test, all_pre_rr_test, all_post_rr_test, all_avg_rr_test = load_data_from_csv(os.path.join('data', 'ecg_test.csv'))
    labels_test = labels_test.astype(int)

    test_dataset = ECGDataset(padded_beats_test, labels_test, all_pre_rr_test, all_post_rr_test, all_avg_rr_test)
    test_loader = DataLoader(test_dataset, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.")

    input_size = 128
    rr_feature_size = 3
    num_classes = 4

    model = CNN_LSTM_Model(input_size=input_size, rr_feature_size=rr_feature_size, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(os.path.join('checkpoints', 'best_model.pth'), map_location=device))

    model.eval()

    correct_predictions = 0
    total_predictions = 0
    test_loss = 0.0

    all_preds = []
    all_labels = []

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels, rr_features in test_loader:
            inputs, labels, rr_features = inputs.to(device), labels.to(device), rr_features.to(device)

            # Forward pass
            outputs = model(inputs, rr_features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = correct_predictions.double() / total_predictions

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f"Class {i}" for i in range(conf_matrix.shape[0])],
                                columns=[f"Class {i}" for i in range(conf_matrix.shape[1])])

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    class_accuracy_df = pd.DataFrame({
        "Class": [f"Class {i}" for i in range(len(class_accuracy))],
        "Accuracy": class_accuracy
    })
    print("\nPer-Class Accuracy:")
    print(class_accuracy_df.to_string(index=False))

    report = classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(conf_matrix.shape[0])])
    print("\nClassification Report:")
    print(report)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')