import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, rr_feature_size, num_classes, lstm_hidden_size=128, lstm_layers=2):
        super(CNN_LSTM_Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Batch Normalization and Pooling
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)

        # Fully Connected Layers after LSTM output and RR features concatenation
        self.fc1 = nn.Linear(lstm_hidden_size * 2 + rr_feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, rr_features):
        x = self.pool(self.batchnorm1(F.relu(self.conv1(x))))
        x = self.pool(self.batchnorm2(F.relu(self.conv2(x))))
        x = self.pool(self.batchnorm3(F.relu(self.conv3(x))))

        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]

        x, _ = self.lstm(x)

        x = x[:, -1, :]  # [batch_size, lstm_hidden_size * 2]

        x = torch.cat((x, rr_features), dim=1)  # [batch_size, lstm_hidden_size * 2 + rr_feature_size]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x