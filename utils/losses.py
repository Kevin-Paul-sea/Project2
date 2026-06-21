"""
Loss functions used in CIFAR-10 experiments.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        selected_log_prob = log_prob.gather(1, targets.view(-1, 1)).squeeze(1)
        selected_prob = prob.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = -((1.0 - selected_prob) ** self.gamma) * selected_log_prob
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(name="cross_entropy", label_smoothing=0.0, focal_gamma=2.0):
    name = str(name).lower()
    if name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss()
    if name in ("label_smoothing", "smooth_ce"):
        return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if name == "focal":
        return FocalLoss(gamma=focal_gamma)
    raise ValueError(f"unknown loss: {name}")
