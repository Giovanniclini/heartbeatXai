from utils._utils import load_data_from_csv
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from dataset.ECGDatasetHandler import ECGDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import CNN_LSTM_Model
from model.focal_loss import FocalLoss
import os

if __name__ == '__main__':

    # Load training and test data
    all_padded_resampled_beats, all_padded_resampled_labels, all_resampled_pre_rr, all_resampled_post_rr, all_resampled_avg_rr = load_data_from_csv(os.path.join('data', 'ecg_training.csv'))

    # Combine the ECG beats with the RR features for the split
    X_combined = np.hstack((all_padded_resampled_beats,
                            all_resampled_pre_rr.reshape(-1, 1),
                            all_resampled_post_rr.reshape(-1, 1),
                            all_resampled_avg_rr.reshape(-1, 1)))

    # Split the data into training and validation sets
    X_train_combined, X_val_combined, y_train, y_val = train_test_split(X_combined, all_padded_resampled_labels, test_size=0.2, random_state=42, stratify=all_padded_resampled_labels)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

    # Separate the resampled data back into ECG beats and RR features
    X_train_resampled_beats = X_train_resampled[:, :-3] 
    X_train_resampled_pre_rr = X_train_resampled[:, -3]
    X_train_resampled_post_rr = X_train_resampled[:, -2]
    X_train_resampled_avg_rr = X_train_resampled[:, -1]

    # Similarly, separate the validation set into ECG beats and RR features
    X_val_beats = X_val_combined[:, :-3]
    X_val_pre_rr = X_val_combined[:, -3]
    X_val_post_rr = X_val_combined[:, -2]
    X_val_avg_rr = X_val_combined[:, -1]

    # Create datasets including the RR features
    train_dataset = ECGDataset(X_train_resampled_beats, y_train_resampled,
                            X_train_resampled_pre_rr, X_train_resampled_post_rr, X_train_resampled_avg_rr)

    val_dataset = ECGDataset(X_val_beats, y_val,
                            X_val_pre_rr, X_val_post_rr, X_val_avg_rr)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train loader batch size: {batch_size}, total batches: {len(train_loader)}')
    print(f'Validation loader batch size: {batch_size}, total batches: {len(val_loader)}')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.")

    # Initialize model, criterion, and optimizer
    input_size = 128
    rr_feature_size = 3
    num_classes = 3

    model = CNN_LSTM_Model(input_size=input_size, rr_feature_size=rr_feature_size, num_classes=num_classes)
    model.to(device)

    criterion = FocalLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.0001)

    print(f'Model initialized and moved to {device}')
    print(f'Optimizer: {optimizer}')
    print(f'Scheduler: {scheduler}')

    num_epochs = 100
    best_val_acc = 0.0
    patience = 10
    epochs_no_improve = 0
    early_stop = False
    model_save_path = os.path.join('checkpoints', 'best_model.pth')

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping")
            break

        model.train()
        running_loss = 0.0
        correct_predictions = 0

        for i, (ecg_signals, labels, rr_features) in enumerate(train_loader):
            ecg_signals, labels, rr_features = ecg_signals.to(device), labels.to(device), rr_features.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(ecg_signals, rr_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0

        with torch.no_grad():
            for ecg_signals, labels, rr_features in val_loader:
                ecg_signals, labels, rr_features = ecg_signals.to(device), labels.to(device), rr_features.to(device)

                outputs = model(ecg_signals, rr_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val_predictions += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader)
        val_acc = correct_val_predictions.double() / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with validation accuracy: {val_acc:.4f}")
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping triggered")
            early_stop = True