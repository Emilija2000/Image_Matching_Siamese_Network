import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: "Dimensionality Reduction by Learning an Invariant Mapping"
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        temp = torch.matmul((1-label).T*1.,torch.pow(distance, 2)) + torch.matmul((label).T*1., torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        loss_contrastive = temp/label.shape[0]
        return loss_contrastive