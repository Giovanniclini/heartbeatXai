import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, beats, labels, pre_rr, post_rr, avg_rr):
        """
        Initializes the ECGDataset.

        Args:
            beats (list or np.array): A list or array of ECG beat signals.
            labels (list or np.array): A list or array of labels corresponding to each beat.
            pre_rr (list or np.array): A list or array of Pre-RR interval features.
            post_rr (list or np.array): A list or array of Post-RR interval features.
            avg_rr (list or np.array): A list or array of Average RR interval features.
        """
        self.beats = beats
        self.labels = labels
        self.pre_rr = pre_rr
        self.post_rr = post_rr
        self.avg_rr = avg_rr

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        ecg_signal = torch.tensor(self.beats[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        # Combine RR features into a single tensor
        rr_features = torch.tensor([self.pre_rr[idx], self.post_rr[idx], self.avg_rr[idx]], dtype=torch.float32)

        return ecg_signal, label, rr_features
