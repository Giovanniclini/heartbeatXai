import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PreprocessingFNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(PreprocessingFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
class LSTMSubmodule(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm1(x_packed)
        out_packed, _ = self.lstm2(out_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])

class FC_submodule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FC_submodule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)

class GRNN(nn.Module):
    def __init__(self, input_size_lstm, hidden_size1, hidden_size2, output_size_lstm, input_size_fc, hidden_size_fc):
        super(GRNN, self).__init__()

        self.lstm_submodule = LSTMSubmodule(input_size_lstm, hidden_size1, hidden_size2, output_size_lstm)
        self.fc_submodule = FC_submodule(input_size_fc, hidden_size_fc, 10)
        self.fc_combined = nn.Linear(output_size_lstm + 10, 4)

    def forward(self, x1, x2, lengths):
        lengths = lengths.cpu().long()
        out1 = self.lstm_submodule(x1, lengths)
        out2 = self.fc_submodule(x2.unsqueeze(0).T)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc_combined(out)
        return out
