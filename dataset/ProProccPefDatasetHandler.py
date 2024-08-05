import torch
from torch.utils.data import Dataset

class PreProcess_ECGDataset(Dataset):
    def __init__(self, pef_labels, rr_features):

        self.pef_labels = pef_labels
        self.rr_features = rr_features

    def __len__(self):
        return len(self.pef_labels)

    def __getitem__(self, idx):
        pef_label = torch.tensor(self.pef_labels[idx], dtype=torch.long)
        rr_features = torch.tensor(self.rr_features[idx], dtype=torch.float32)

        return rr_features, pef_label