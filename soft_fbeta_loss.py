import torch
import torch.nn as nn

class SoftF1Loss(nn.Module):
    def __init__(self, beta=1.0, eps=1e-7):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        beta2 = self.beta ** 2
        soft_f1 = (1 + beta2) * tp / (
            (1 + beta2) * tp + beta2 * fn + fp + self.eps
        )
        loss = 1 - soft_f1
        return loss

def make_weighted_bce_loss(pos_weight):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    def loss_fn(logits, targets):
        targets = targets.view(-1, 1)
        logits = logits.view(-1, 1)
        return bce(logits, targets)
    return loss_fn
