import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else torch.tensor(1.0)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure alpha and targets are on the same device
        if self.alpha.device != targets.device:
            self.alpha = self.alpha.to(targets.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the class-specific alpha values for each example in the batch
        alpha_t = self.alpha[targets]
        
        pt = torch.exp(-ce_loss)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss