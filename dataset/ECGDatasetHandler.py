import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, beats, labels, pef_labels):
        self.beats = beats
        self.labels = labels
        self.pef_labels = pef_labels

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        ecg_signal = torch.tensor(self.beats[idx], dtype=torch.float32)
        beat_label = torch.tensor(self.labels[idx], dtype=torch.int64)
        pef_label = torch.tensor(self.pef_labels[idx], dtype=torch.float32)

        return {
            'ecg_signal': ecg_signal,
            'beat_label': beat_label,
            'pef_label': pef_label
        }